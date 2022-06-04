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

from numpy import result_type
from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
from dlk.utils.logger import Logger
import os
import pandas as pd

logger = Logger.get_logger()

@subprocessor_config_register('word_mask')
class WordMaskConfig(BaseConfig):
    """Config for WordMask

    Config Example:
        >>> {
        >>>     "_name": "word_mask",
        >>>     "config": {
        >>>         "train":{
        >>>             "input_map": {  // without necessery, don't change this
        >>>                 "word_ids": "word_ids",
        >>>                 "offsets": "offsets",
        >>>                 "entities_info": "entities_info",
        >>>             },
        >>>             "data_set": {                   // for different stage, this processor will process different part of data
        >>>                 "train": ['train', 'valid', 'test'],
        >>>                 "predict": ['predict'],
        >>>                 "online": ['online']
        >>>             },
        >>>             "output_map": {
        >>>                 "labels": "labels",
        >>>             },
        >>>         },
        >>>         "predict": "train",
        >>>         "online": "train",
        >>>     }
        >>> }
    """
    def __init__(self, stage, config: Dict):

        super(WordMaskConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.word_ids = self.config['input_map']['word_ids']
        self.offsets = self.config['input_map']['offsets']
        self.entities_info = self.config['input_map']['entities_info']
        self.clean_droped_entity = self.config['clean_droped_entity']
        self.drop = self.config['drop']
        self.start_label = self.config['start_label']
        self.end_label = self.config['end_label']
        self.output_labels = self.config['output_map']['labels']
        self.entity_priority = {entity: priority for priority, entity in enumerate(self.config['entity_priority'])}
        self.priority_trigger = self.config['priority_trigger']
        self.post_check(self.config, used=[
            "input_map",
            "data_set",
            "drop",
            "output_map",
        ])


@subprocessor_register('word_mask')
class WordMask(ISubProcessor):
    """
    Relabel the json data to bio
    """

    def __init__(self, stage: str, config: WordMaskConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'word_mask' at stage {self.stage}")
            return

    def process(self, data: Dict)->Dict:
        """WordMask Entry

        Args:
            data: Dict

        Returns: 
            
            relabeled data

        """

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do word_mask on it.')
                continue
            data_set = data['data'][data_set_name]
            if os.environ.get('DISABLE_PANDAS_PARALLEL', 'false') != 'false':
                data_set[[self.config.output_labels,
                    self.config.entities_info
                ]] = data_set.parallel_apply(self.relabel, axis=1, result_type='expand')
            else:
                data_set[[self.config.output_labels,
                    self.config.entities_info
                ]] = data_set.apply(self.relabel, axis=1, result_type='expand')

        return data
