# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hjson
import pickle as pkl
import os
import json
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlk.utils.logger import Logger
import torch
from dlk.utils.vocab import Vocabulary

logger = Logger.get_logger()


@postprocessor_config_register('txt_cls')
class TxtClsPostProcessorConfig(IPostProcessorConfig):
    """Config for TxtClsPostProcessor

    Config Example:
        >>> {
        >>>     "_name": "txt_cls",
        >>>     "config": {
        >>>         "meta": "*@*",
        >>>         "meta_data": {
        >>>             "label_vocab": 'label_vocab',
        >>>         },
        >>>         "input_map": {
        >>>             "logits": "logits",
        >>>             "label_ids": "label_ids"
        >>>             "_index": "_index",
        >>>         },
        >>>         "origin_input_map": {
        >>>             "sentence": "sentence",
        >>>             "sentence_a": "sentence_a", // for pair
        >>>             "sentence_b": "sentence_b",
        >>>             "uuid": "uuid"
        >>>         },
        >>>         "save_root_path": ".",  //save data root dir
        >>>         "top_k": 1, //the result return top k result
        >>>         "data_type": "single", //single or pair
        >>>         "save_path": {
        >>>             "valid": "valid",  // relative dir for valid stage
        >>>             "test": "test",    // relative dir for test stage
        >>>         },
        >>>         "start_save_step": 0,  // -1 means the last
        >>>         "start_save_epoch": -1,
        >>>     }
        >>> }
    """

    def __init__(self, config: Dict):
        super(TxtClsPostProcessorConfig, self).__init__(config)

        self.data_type = self.config['data_type']
        assert self.data_type in {'single', 'pair'}
        if self.data_type == 'pair':
            self.sentence_a = self.origin_input_map['sentence_a']
            self.sentence_b = self.origin_input_map['sentence_b']
        else:
            self.sentence = self.origin_input_map['sentence']
        self.uuid = self.origin_input_map['uuid']

        self.logits = self.input_map['logits']
        self.label_ids = self.input_map['label_ids']
        self._index = self.input_map['_index']
        self.top_k = self.config['top_k']
        if isinstance(self.config['meta'], str):
            meta = pkl.load(open(self.config['meta'], 'rb'))
        else:
            raise PermissionError("You must provide meta data for txt_cls postprocess.")
        trace_path = []
        trace_path_str = self.config['meta_data']['label_vocab']
        if trace_path_str and trace_path_str.strip()!='.':
            trace_path = trace_path_str.split('.')
        self.label_vocab = meta
        for trace in trace_path:
            self.label_vocab = self.label_vocab[trace]
        self.label_vocab = Vocabulary.load(self.label_vocab)
        self.save_path = self.config['save_path']
        self.save_root_path = self.config['save_root_path']
        self.start_save_epoch = self.config['start_save_epoch']
        self.start_save_step = self.config['start_save_step']
        self.post_check(self.config, used=[
            "meta",
            "meta_data",
            "input_map",
            "origin_input_map",
            "save_root_path",
            "top_k",
            "data_type",
            "save_path",
            "start_save_step",
            "start_save_epoch",
        ])


@postprocessor_register('txt_cls')
class TxtClsPostProcessor(IPostProcessor):
    """postprocess for text classfication"""
    def __init__(self, config: TxtClsPostProcessorConfig):
        super(TxtClsPostProcessor, self).__init__()
        self.config = config
        self.label_vocab = self.config.label_vocab

    def do_predict(self, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict)->List:
        """Process the model predict to human readable format

        Args:
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns: 
            all predicts

        """
        results = []
        for outputs in list_batch_outputs:
            logits = outputs[self.config.logits].detach()
            assert len(logits.shape) == 2
            # predict_indexes = list(torch.argmax(logits, 1))
            indexes = list(outputs[self.config._index])

            if self.config.label_ids in outputs:
                label_ids = outputs[self.config.label_ids]
            else:
                label_ids = [None] * len(indexes)
            for i, (one_logits, index, label_id) in enumerate(zip(logits, indexes, label_ids)):
                one_ins = {}
                one_logits = torch.softmax(one_logits, -1)
                one_origin = origin_data.iloc[int(index)]
                if self.config.data_type == 'single':
                    sentence = one_origin[self.config.sentence]
                    one_ins['sentence'] = sentence
                else:
                    sentence_a = one_origin[self.config.sentence_a]
                    one_ins['sentence_a'] = sentence_a
                    sentence_b = one_origin[self.config.sentence_b]
                    one_ins['sentence_b'] = sentence_b
                uuid = one_origin[self.config.uuid]
                label_values, label_indeies = torch.topk(one_logits, self.config.top_k, dim=-1)
                predict = {}
                for i, (label_value, label_index) in enumerate(zip(label_values, label_indeies)):
                    label_name = self.label_vocab.get_word(label_index)
                    predict[i] = [label_name, float(label_value)]
                ground_truth = []
                if label_id is not None and label_id.shape:
                    for one_label_id in label_id:
                        ground_truth.append(self.label_vocab.get_word(one_label_id))
                elif label_id:
                    ground_truth.append(self.label_vocab.get_word(label_id))
                one_ins['uuid'] = uuid
                one_ins['labels'] = ground_truth
                one_ins['predicts'] = predict
                one_ins['predict_extend_return'] = self.gather_predict_extend_data(outputs, i, self.config.predict_extend_return)
                results.append(one_ins)
        return results

    def do_calc_metrics(self, predicts: List, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict)->Dict:
        """calc the scores use the predicts or list_batch_outputs

        Args:
            predicts: list of predicts
                
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns: 
            the named scores, acc

        """
        right_num = 0
        for one_ins in predicts:
            labels = one_ins['labels']
            assert len(labels) == 1, "We currently is not support multi label in txt_cls postprocess"
            label = labels[0]
            one_predicts = one_ins['predicts']
            predict_label, predict_value = one_predicts[0] # the first predict
            if label == predict_label:
                right_num += 1
        real_name = self.loss_name_map(stage)
        return {f'{real_name}_acc': right_num/len(predicts)}

    def do_save(self, predicts: List, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict, save_condition: bool=False):
        """save the predict when save_condition==True

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }
            save_condition: True for save, False for depend on rt_config

        Returns: 
            None

        """
        if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
            self.config.start_save_step = rt_config.get('total_steps', 0) - 1
            self.config.start_save_epoch = rt_config.get('total_epochs', 0) - 1
        if not save_condition and (rt_config['current_step']>=self.config.start_save_step or rt_config['current_epoch']>=self.config.start_save_epoch):
            save_condition = True
        if save_condition:
            save_path = os.path.join(self.config.save_root_path, self.config.save_path.get(stage, ''))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            if "current_step" in rt_config:
                save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}_predict.json")
            else:
                save_file = os.path.join(save_path, 'predict.json')
            logger.info(f"Save the {stage} predict data at {save_file}")
            json.dump(predicts, open(save_file, 'w'), indent=4, ensure_ascii=False)

