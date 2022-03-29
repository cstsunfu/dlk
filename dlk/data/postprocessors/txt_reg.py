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
import torch
from typing import Union, Dict, Any, List
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlk.utils.logger import Logger
from dlk.utils.vocab import Vocabulary
import torchmetrics

logger = Logger.get_logger()


@postprocessor_config_register('txt_reg')
class TxtRegPostProcessorConfig(IPostProcessorConfig):
    """Config for TxtRegPostProcessor

    Config Example:
        >>> {
        >>>     "_name": "txt_reg",
        >>>     "config": {
        >>>         "input_map": {
        >>>             "logits": "logits",
        >>>             "values": "values",
        >>>             "_index": "_index",
        >>>         },
        >>>         "origin_input_map": {
        >>>             "sentence": "sentence",
        >>>             "sentence_a": "sentence_a", // for pair
        >>>             "sentence_b": "sentence_b",
        >>>             "uuid": "uuid"
        >>>         },
        >>>         "data_type": "single", //single or pair
        >>>         "save_root_path": ".",  //save data root dir
        >>>         "save_path": {
        >>>             "valid": "valid",  // relative dir for valid stage
        >>>             "test": "test",    // relative dir for test stage
        >>>         },
        >>>         "log_reg": false, // whether logistic regression
        >>>         "start_save_step": 0,  // -1 means the last
        >>>         "start_save_epoch": -1,
        >>>     }
        >>> }
    """

    def __init__(self, config: Dict):
        super(TxtRegPostProcessorConfig, self).__init__(config)

        self.data_type = self.config['data_type']
        assert self.data_type in {'single', 'pair'}
        if self.data_type == 'pair':
            self.sentence_a = self.origin_input_map['sentence_a']
            self.sentence_b = self.origin_input_map['sentence_b']
        else:
            self.sentence = self.origin_input_map['sentence']
        self.uuid = self.origin_input_map['uuid']
        self.log_reg = self.config['log_reg']

        self.value = self.input_map['values']
        self.logits = self.input_map['logits']
        self._index = self.input_map['_index']
        self.save_path = self.config['save_path']
        self.save_root_path = self.config['save_root_path']
        self.start_save_epoch = self.config['start_save_epoch']
        self.start_save_step = self.config['start_save_step']
        self.post_check(self.config, used=[
            "input_map",
            "origin_input_map",
            "save_root_path",
            "save_path",
            "data_type",
            "start_save_step",
            "start_save_epoch",
            "log_reg",
        ])


@postprocessor_register('txt_reg')
class TxtRegPostProcessor(IPostProcessor):
    """text regression postprocess"""
    def __init__(self, config: TxtRegPostProcessorConfig):
        super(TxtRegPostProcessor, self).__init__()
        self.config = config

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
            if self.config.log_reg:
                logits = torch.sigmoid(logits)
            assert len(logits.shape) == 2
            # predict_indexes = list(torch.argmax(logits, 1))
            indexes = list(outputs[self.config._index])

            if self.config.value in outputs:
                values = outputs[self.config.value]
            else:
                values = [0.0] * len(indexes)
            for i, (one_logits, index, value) in enumerate(zip(logits, indexes, values)):
                one_ins = {}
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
                one_ins['uuid'] = uuid
                one_ins['values'] = [float(value)]
                one_ins['predict_values'] = [float(one_logits)]
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
            the named scores

        """
        return {}

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

