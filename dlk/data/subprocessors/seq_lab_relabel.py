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

@subprocessor_config_register('seq_lab_relabel')
class SeqLabRelabelConfig(BaseConfig):
    """Config for SeqLabRelabel

    Config Example:
        >>> {
        >>>     "_name": "seq_lab_relabel",
        >>>     "config": {
        >>>         "train":{ //train、predict、online stage config,  using '&' split all stages
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
        >>>             "drop": "shorter", //'longer'/'shorter'/'none', if entities is overlap, will remove by rule
        >>>             "start_label": "S",
        >>>             "end_label": "E",
        >>>             "clean_droped_entity": true, // after drop entity for training, whether drop the entity for calc metrics, default is true, this only works when the drop != 'none'
        >>>             "entity_priority": [],
        >>>             //"entity_priority": ['Product'],
        >>>             "priority_trigger": 1, // if the overlap entity abs(length_a - length_b)<=priority_trigger, will trigger the entity_priority strategy
        >>>         }, //3
        >>>         "predict": "train",
        >>>         "online": "train",
        >>>     }
        >>> }
    """
    def __init__(self, stage, config: Dict):

        super(SeqLabRelabelConfig, self).__init__(config)
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
            "start_label",
            "end_label",
            "clean_droped_entity",
            "entity_priority",
            "priority_trigger",
        ])


@subprocessor_register('seq_lab_relabel')
class SeqLabRelabel(ISubProcessor):
    """
    Relabel the json data to bio
    """

    def __init__(self, stage: str, config: SeqLabRelabelConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'seq_lab_relabel' at stage {self.stage}")
            return

    def process(self, data: Dict)->Dict:
        """SeqLabRelabel Entry

        Args:
            data: Dict

        Returns: 
            
            relabeled data

        """

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do seq_lab_relabel on it.')
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

    def find_position_in_offsets(self, position: int, offset_list: List, sub_word_ids: List, start: int, end: int, is_start: bool=False):
        """find the sub_word index which the offset_list[index][0]<=position<offset_list[index][1]

        Args:
            position: position
            offset_list: list of all tokens offsets
            sub_word_ids: word_ids from tokenizer
            start: start search index
            end: end search index
            is_start: is the position is the start of target token, if the is_start==True and cannot find return -1

        Returns: 
            the index of the offset which include position

        """
        while start<end:
            if sub_word_ids[start] is None:
                start += 1
            elif position>=offset_list[start][0] and position<offset_list[start][1]:
                return start
            elif position<offset_list[start][0]:
                if start == 1 and offset_list[0] == [0, 0]:
                    return 1
                if is_start:
                    return -1
                else:
                    return start - 1
            else:
                start += 1
        return -1

    def relabel(self, one_ins: pd.Series):
        """make token label, if use the first piece label please use the 'seq_lab_firstpiece_relabel'

        Args:
            one_ins: include sentence, entity_info, offsets

        Returns: 
            labels(labels for each subtoken)

        """
        pre_clean_entities_info = one_ins[self.config.entities_info]
        pre_clean_entities_info.sort(key=lambda x: x['start'])
        offsets = one_ins[self.config.offsets]
        sub_word_ids = one_ins[self.config.word_ids]
        if not sub_word_ids:
            logger.warning(f"entity_info: {pre_clean_entities_info}, offsets: {offsets} ")

        entities_info = []
        pre_end = -1
        pre_length = 0
        pre_label = ''
        for entity_info in pre_clean_entities_info:
            assert len(entity_info['labels']) == 1, f"currently we just support one label for one entity"
            if entity_info['start']<pre_end: # if overlap will remove one
                if self.config.drop == 'none':
                    pass
                elif abs(entity_info['end'] - entity_info['start'] - pre_length) <= self.config.priority_trigger:
                    pre_label_order = self.config.entity_priority.get(pre_label, 1e9)
                    label_order = self.config.entity_priority.get(entity_info['labels'][0], 1e9)
                    if label_order<pre_label_order:
                        entities_info.pop()
                    else:
                        continue
                elif self.config.drop == 'shorter':
                    if entity_info['end'] - entity_info['start'] > pre_length:
                        entities_info.pop()
                    else:
                        continue
                elif self.config.drop =='longer':
                    if entity_info['end'] - entity_info['start'] < pre_length:
                        entities_info.pop()
                    else:
                        continue
                else:
                    raise PermissionError(f"The drop method must in 'none'/'shorter'/'longer'")
                pre_label = entity_info['labels'][0]
            entities_info.append(entity_info)
            pre_end = entity_info['end']
            pre_length = entity_info['end'] - entity_info['start']

        cur_token_index = 0
        offset_length = len(offsets)
        sub_labels = []
        for entity_info in entities_info:
            start_token_index = self.find_position_in_offsets(entity_info['start'], offsets, sub_word_ids, cur_token_index, offset_length, is_start=True)
            if start_token_index == -1:
                logger.warning(f"cannot find the entity_info : {entity_info}, offsets: {offsets} ")
                continue
            for _ in range(start_token_index-cur_token_index):
                sub_labels.append('O')
            end_token_index = self.find_position_in_offsets(entity_info['end']-1, offsets, sub_word_ids, start_token_index, offset_length)
            assert end_token_index != -1, f"entity_info: {entity_info}, offsets: {offsets}"
            sub_labels.append("B-"+entity_info['labels'][0])
            for _ in range(end_token_index-start_token_index):
                sub_labels.append("I-"+entity_info['labels'][0])
            cur_token_index = end_token_index + 1
        assert cur_token_index<=offset_length
        for _ in range(offset_length-cur_token_index):
            sub_labels.append('O')

        if sub_word_ids[0] is None:
            sub_labels[0] = self.config.start_label

        if sub_word_ids[offset_length-1] is None:
            sub_labels[offset_length-1] = self.config.end_label

        if len(sub_labels)!= offset_length:
            logger.error(f"{len(sub_labels)} vs {offset_length}")
            for i in one_ins:
                logger.error(f"{i}")
            raise PermissionError

        if not self.config.clean_droped_entity:
            entities_info = one_ins[self.config.entities_info]
        return sub_labels, entities_info
