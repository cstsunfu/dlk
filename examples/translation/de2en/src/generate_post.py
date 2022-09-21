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
from tokenizers import Tokenizer
from dlk.utils.vocab import Vocabulary
from dlk.utils.io import open
import torchmetrics

logger = Logger.get_logger()


@postprocessor_config_register('token_generate')
class TokenGeneratePostProcessorConfig(IPostProcessorConfig):
    """Config for TokenGeneratePostProcessor

    Config Example:
        >>> {
        >>>     "_name": "token_generate",
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
        super(TokenGeneratePostProcessorConfig, self).__init__(config)

        # self.post_check(self.config, used=[
        #     "input_map",
        #     "origin_input_map",
        #     "save_root_path",
        #     "save_path",
        #     "data_type",
        #     "start_save_step",
        #     "start_save_epoch",
        #     "log_reg",
        # ])


@postprocessor_register('token_generate')
class TokenGeneratePostProcessor(IPostProcessor):
    """text regression postprocess"""
    def __init__(self, config: TokenGeneratePostProcessorConfig):
        super(TokenGeneratePostProcessor, self).__init__()
        self.config = config

        self.tokenizer = Tokenizer.from_file('./bart/tokenizer.json')

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
        print('\n---')
        print(len(list_batch_outputs))
        outputs = list_batch_outputs[0]
        # print(outputs)
        print(f"Source:\n", self.tokenizer.decode(list(outputs['target_ids'][0]), skip_special_tokens=False))
        # print(outputs['encoder_input_ids'].shape)
        # print(outputs['decoder_input_ids'].shape)
        print(f"Generate 1:\n", self.tokenizer.decode(list(outputs['generated'][0][0]['tokens']), skip_special_tokens=False))
        print(f"Generate 2:\n", self.tokenizer.decode(list(outputs['generated'][0][1]['tokens']), skip_special_tokens=False))
        print(f"Generate 3:\n", self.tokenizer.decode(list(outputs['generated'][0][2]['tokens']), skip_special_tokens=False))
        # print(list(outputs['generated'][0][0]))
        print('---')
        pass
        # return results

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

        pass

