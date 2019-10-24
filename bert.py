from __future__ import absolute_import, division, print_function

import collections
import logging
import math
from inference_optimization import OptimizedModel

import numpy as np
import torch
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  DistilBertConfig,
                                  DistilBertForQuestionAnswering,
                                  DistilBertTokenizer,
                                  RobertaConfig,
                                  RobertaForQuestionAnswering,
                                  RobertaTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import (get_answer, input_to_squad_example,
                   squad_examples_to_features, to_list)


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

MODELS = {'roberta': (RobertaConfig, RobertaTokenizer, RobertaForQuestionAnswering),
          'distilbert': (DistilBertConfig, DistilBertTokenizer, DistilBertForQuestionAnswering),
          'bert': (BertConfig, BertTokenizer, BertForQuestionAnswering)}

class QA:

    def __init__(self,model_path: str, xla=False, fp16=False, onnx=False, export_onnx=False, sess=None, vsl=False, onnx_path=None):
        self.max_seq_length = 384
        self.doc_stride = 256
        self.do_lower_case = False
        self.max_query_length = 64
        self.n_best_size = 3
        self.max_answer_length = 50
        self.vsl = vsl
        if sess is not None:
            self.use_tf = True
            self.model = sess
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.do_lower_case)
        else:
            self.use_tf = False
            self.model, self.tokenizer = self.load_model(model_path)
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            # inputs = {'input_ids': torch.ones((1, 3), dtype=torch.long).cuda(),
            #           'attention_mask': torch.ones((1, 3), dtype=torch.long).cuda(),
            #           'token_type_ids': torch.ones((1, 3), dtype=torch.long).cuda()
            #           }
            # self.device = 'cpu'
            self.use_xla = xla
            use_fp16 = fp16
            use_trt = False
            self.use_onnx_runtime = onnx
            self.model = self.model.to(self.device)

            optimizer = OptimizedModel(self.model, self.device)
            self.model = optimizer.optimize(use_xla=self.use_xla,
                                            use_fp16=use_fp16,
                                            use_trt=use_trt,
                                            export_onnx=export_onnx,
                                            use_onnx_runtime=self.use_onnx_runtime,
                                            onnx_path=onnx_path)

    def load_model(self,model_path: str,do_lower_case=False):
        model_type = 'bert'
        config, tokenizer, model = MODELS[model_type]

        if '/' in model_path:
            config = config.from_pretrained(model_path + "/config.json")
            tokenizer = tokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
            model = model.from_pretrained(model_path, from_tf=False, config=config)
        else:
            tokenizer = tokenizer.from_pretrained(model_path)
            model = model.from_pretrained(model_path)
        return model, tokenizer
    
    def predict(self,passages :list,question :str):
        logging.info('Receiving request')
        examples = input_to_squad_example(passages,question)
        features = squad_examples_to_features(examples,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length, vsl=self.vsl)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=len(features))
        all_results = []
        for batch in eval_dataloader:
            if not self.use_tf:
                if not self.use_onnx_runtime:
                    batch = tuple(t.to(self.device) for t in batch)
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
            with torch.no_grad():
                if self.use_tf:
                    # inputs = {'input_ids:0': to_numpy(batch[0]),
                    #           'attention_mask:0': to_numpy(batch[1]),
                    #           'token_type_ids:0': to_numpy(batch[2])
                    #           }
                    inputs = {'input_ids:0': to_numpy(batch[0]),
                              'input_mask:0': to_numpy(batch[1]),
                              'segment_ids:0': to_numpy(batch[2])
                              }
                    example_indices = batch[3]
                    # start_logits, end_logits = self.model.run(['Squeeze_49:0', 'Squeeze_50:0'], feed_dict=inputs)
                    start_logits, end_logits = self.model.run(['start_logits:0', 'end_logits:0'], feed_dict=inputs)
                    outputs = [start_logits, end_logits]
                elif self.use_onnx_runtime:
                    inputs = {self.model.get_inputs()[0].name: to_numpy(batch[0]),
                              self.model.get_inputs()[1].name: to_numpy(batch[1]),
                              self.model.get_inputs()[2].name: to_numpy(batch[2])
                              }
                    output_names = [self.model.get_outputs()[0].name,
                                    self.model.get_outputs()[1].name
                                    ]
                    example_indices = batch[3]
                    outputs = self.model.run(output_names, inputs)
                else:
                    inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1],
                              'input' if self.use_xla else 'token_type_ids': batch[2]
                              }
                    example_indices = batch[3]
                    outputs = self.model(**inputs)
                logging.info('Processed request answer: {}'.format(outputs))

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
        answers = get_answer(examples,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answers
