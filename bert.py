from __future__ import absolute_import, division, print_function

import collections
import logging
import math
import tensorflow
# import tensorrt as trt
# import onnx_tensorrt.backend as backend


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



class QA:

    def __init__(self,model_path: str):
        self.max_seq_length = 384
        self.doc_stride = 256
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 3
        self.max_answer_length = 50
        self.model, self.tokenizer = self.load_model(model_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        inputs = {'input_ids': torch.ones((1, 3), dtype=torch.long).cuda(),
                  'attention_mask': torch.ones((1, 3), dtype=torch.long).cuda(),
                  'token_type_ids': torch.ones((1, 3), dtype=torch.long).cuda()
                  }
        x = torch.ones([1,384], dtype=torch.long).to(self.device)
        use_xla = False
        use_fp16 = False
        use_trt = False
        self.vsl = False
        self.model = self.model.to(self.device)

        if use_fp16:
            from pytorch_transformers import AdamW
            from apex import amp
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters()],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=0., eps=0.)
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level='O2')
        if use_xla:
            # cannot use xla with trt
            with torch.jit.optimized_execution(True):
                self.model = torch.jit.trace(self.model, (x), check_trace=True)
            # self.model = torch.jit.script(self.model)
        # elif use_trt:
        #     from torch2trt import torch2trt
            # onnx_path = "/home/volker/workspace/gitrepo/BERT-SQuAD/spanbert_qa.onnx"
            # torch.onnx.export(self.model, x, onnx_path, export_params=True)
            # model = onnx.load(onnx_path)
            # engine = backend.prepare(model, device='CUDA:1')
            # input_data = np.random.random(size=(1, 1)).astype(np.int32)
            # output_data = engine.run(input_data)[0]
            # model_trt = torch2trt(self.model, [x,x,x], max_batch_size=1, max_workspace_size=4e9, log_level=trt.Logger.WARNING)
        self.model.eval()

    def load_model(self,model_path: str,do_lower_case=False):
        MODELS = {'roberta': (RobertaConfig, RobertaTokenizer, RobertaForQuestionAnswering),
                  'distilbert': (DistilBertConfig, DistilBertTokenizer, DistilBertForQuestionAnswering),
                  'bert': (BertConfig, BertTokenizer, BertForQuestionAnswering)}
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
    
    def predict(self,passage :str,question :str):
        logging.info('Receiving request')
        example = input_to_squad_example(passage,question)
        features = squad_examples_to_features(example,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length, vsl=self.vsl)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)
        all_results = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] #token_type_ids input
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
        answer = get_answer(example,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answer
