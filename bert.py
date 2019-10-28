from __future__ import absolute_import, division, print_function

import collections
import logging
import math
from inference_optimization import OptimizedModel
import os

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

    def __init__(self,model_path: str, use_jit=False, fp16=False, onnx=False, export_onnx=False, sess=None, tf_onnx=False, vsl=False, onnx_path=None):
        self.max_seq_length = 384
        self.doc_stride = 256
        self.do_lower_case = False
        self.max_query_length = 64
        self.n_best_size = 3
        self.max_answer_length = 50
        self.vsl = vsl
        self.use_onnx_runtime = onnx
        if sess is not None:
            self.use_tf = True
            self.tf_onnx = tf_onnx
            self.model = sess
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.do_lower_case)
        else:
            self.use_tf = False
            model, self.tokenizer = self.load_model(model_path)
            # dir_path = os.path.dirname(os.path.realpath(__file__))
            # self.model = torch.jit.load(os.path.join(dir_path, 'traced_model{}.pt'.format('_fp16' if fp16 else '')))
            # self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.do_lower_case)
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            # self.device = 'cpu'
            self.use_jit = use_jit
            use_trt = False
            model = model.to(self.device)

            optimizer = OptimizedModel(model, self.device)
            self.model = optimizer.optimize(use_jit=self.use_jit,
                                            use_fp16=fp16,
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
        examples = input_to_squad_example(passages,question)
        features = squad_examples_to_features(examples,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length, vsl=self.vsl)
        if not self.use_tf and not self.use_onnx_runtime:
            torch_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
            torch_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
            torch_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
            torch_example_index = torch.arange(torch_input_ids.size(0), dtype=torch.long).to(self.device)
        all_results = []
        if self.use_tf:
            if self.tf_onnx:
                # this is the TF graph converted from ONNX
                inputs = {'input_ids:0': [f.input_ids for f in features],
                          'attention_mask:0': [f.input_mask for f in features],
                          'token_type_ids:0': [f.segment_ids for f in features]
                          }
                start_logits, end_logits = self.model.run(
                    ['Squeeze_49:0', 'Squeeze_50:0'], feed_dict=inputs)
            else:
                # this is the original TF graph
                inputs = {'input_ids:0': [f.input_ids for f in features],
                          'input_mask:0': [f.input_mask for f in features],
                          'segment_ids:0': [f.segment_ids for f in features]
                          }
                start_logits, end_logits = self.model.run(['start_logits:0', 'end_logits:0'], feed_dict=inputs)
            example_indices = np.arange(len(features))
            outputs = [start_logits, end_logits]
        elif self.use_onnx_runtime:
            inputs = {self.model.get_inputs()[0].name: np.array([f.input_ids for f in features]),
                      self.model.get_inputs()[1].name: np.array([f.input_mask for f in features]),
                      self.model.get_inputs()[2].name: np.array([f.segment_ids for f in features])
                      }
            output_names = [self.model.get_outputs()[0].name,
                            self.model.get_outputs()[1].name
                            ]
            example_indices = np.arange(len(features))
            outputs = self.model.run(output_names, inputs)
        else:
            example_indices = torch_example_index
            if self.use_jit:
                outputs = self.model(torch_input_ids, torch_input_mask, torch_segment_ids)
            else:
                with torch.no_grad():
                    inputs = {'input_ids':      torch_input_ids,
                              'attention_mask': torch_input_mask,
                              'token_type_ids': torch_segment_ids
                              }
                    outputs = self.model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
            all_results.append(result)
        answers = get_answer(examples,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answers
