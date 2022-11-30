# Copyright cstsunfu. All rights reserved.
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

from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
from dlk.utils.logger import Logger
import numpy as np
import pandas as pd
import os

logger = Logger.get_logger()

@subprocessor_config_register('span_cls_relabel')
class SpanClsRelabelConfig(BaseConfig):
    default_config = {
        "_name": "span_cls_relabel",
        "config": {
            "train":{
                "input_map": {  
                     "word_ids": "word_ids",
                     "offsets": "offsets",
                     "entities_info": "entities_info",
                },
                "data_set": {                   # for different stage, this processor will process different part of data
                     "train": ['train', 'valid', 'test'],
                     "predict": ['predict'],
                },
                "output_map": {
                     "label_ids": "label_ids",
                     # for span relation extract relabel, deliver should be {"entity_id": {"start": start, "end": end}}, which the start and end should be the index of the token level
                     "processed_entities_info": "processed_entities_info",
                },
                "drop": "none", # 'longer'/'shorter'/'none', if entities is overlap, will remove by rule
                "vocab": "label_vocab", # usually provided by the "token_gather" module
                "entity_priority": [],
                "priority_trigger": 1, # if the overlap entity abs(length_a - length_b)<=priority_trigger, will trigger the entity_priority strategy, otherwise use the drop rule
                "mask_fill": -100,
                "mask_first_sent": False,  # when we use this module to resolve the MRC like SQuAD we should mask the first sentence(quesiotn) for anwering
                "null_to_zero_index": False,  # if cannot find the entity, set to point to the first(zero) index token
                "strict": True, # if strict == True, will drop the unvalid sample
            },
            "extend_train": "train"
        }
    }
    """Config for SpanClsRelabel
    Config Example: 
        default_config
    """
    def __init__(self, stage, config: Dict):

        super(SpanClsRelabelConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.word_ids = self.config['input_map']['word_ids']
        self.offsets = self.config['input_map']['offsets']
        self.mask_fill = self.config['mask_fill']
        self.entities_info = self.config['input_map']['entities_info']
        self.drop = self.config['drop']
        assert self.drop in {'none', 'longer', 'shorter'}
        self.strict = self.config['strict']
        self.vocab = self.config['vocab']
        self.processed_entities_info = self.config['output_map']['processed_entities_info']
        self.output_labels = self.config['output_map']['label_ids']
        self.entity_priority = {entity: priority for priority, entity in enumerate(self.config['entity_priority'])}
        self.priority_trigger = self.config['priority_trigger']
        self.mask_first_sent = self.config['mask_first_sent']
        self.null_to_zero_index = self.config['null_to_zero_index']
        self.post_check(self.config, used=[
            "drop",
            "vocab",
            "mask_first_sent",
            "input_map",
            "data_set",
            "output_map",
            "entity_priority",
            "priority_trigger",
            "null_to_zero_index",
            "mask_first_sent",
            "mask_fill"
        ])


@subprocessor_register('span_cls_relabel')
class SpanClsRelabel(ISubProcessor):
    """
    Relabel the char level entity span to token level and construct matrix
    """

    def __init__(self, stage: str, config: SpanClsRelabelConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.vocab = None
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'span_cls_relabel' at stage {self.stage}")
            return

    def process(self, data: Dict)->Dict:
        """SpanClsRelabel Entry

        Args:
            data: Dict

        Returns: 
            
            relabeled data

        """

        if not self.data_set:
            return data
        # NOTE: only load once, because the vocab should not be changed in same process
        if not self.vocab:
            self.vocab = Vocabulary.load(data[self.config.vocab])
            assert self.vocab.word2idx[self.vocab.unknown] == 0, f"For span_relation_relabel, 'unknown' must be index 0, and other labels as 1...num_label"
            assert not self.vocab.pad, f"For span_relation_relabel, 'pad' must be index 0, and other labels as 1...num_label"

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do span_cls_relabel on it.')
                continue
            data_set = data['data'][data_set_name]
            if os.environ.get('DISABLE_PANDAS_PARALLEL', 'false') != 'false':
                data_set[[self.config.output_labels,
                    self.config.processed_entities_info,
                ]] = data_set.parallel_apply(self.relabel, axis=1, result_type='expand')
            else:
                data_set[[self.config.output_labels,
                    self.config.processed_entities_info,
                ]] = data_set.apply(self.relabel, axis=1, result_type='expand')
            if self.config.strict:
                data_set.dropna(axis=0, inplace=True)
                data_set.reset_index(inplace=True)

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
        """make token label, if use the first piece label please use the 'span_cls_firstpiece_relabel'

        Args:
            one_ins: include sentence, entity_info, offsets

        Returns: 
            labels(labels for each subtoken)
            entities_info
            processed_entities_info: for relation relabal
        """
        pre_clean_entities_info = one_ins[self.config.entities_info]
        if self.config.drop != 'none':
            pre_clean_entities_info.sort(key=lambda x: x['start'])
        offsets = one_ins[self.config.offsets]
        sub_word_ids = one_ins[self.config.word_ids]

        if self.config.mask_first_sent:
            first_start = sub_word_ids.index(0)
            second_start = sub_word_ids[first_start+1:].index(0)
            mask_first_index = first_start + second_start + 2
        else:
            mask_first_index = 0 # if there is only one sentence, set to 0

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
            
        offset_length = len(offsets)

        unknown_id = self.vocab.get_index(self.vocab.unknown)
        mask_matrices = np.full((offset_length, offset_length), self.config.mask_fill, dtype=np.int8)
        mask_matrices = np.tril(mask_matrices, k=-1)

        unknown_matrices = np.full((offset_length, offset_length), unknown_id, dtype=np.int8)
        unknown_matrices = np.triu(unknown_matrices, k=0)

        label_matrices = unknown_matrices + mask_matrices
        if sub_word_ids[0] is None:
            label_matrices[0, :] = self.config.mask_fill
        if sub_word_ids[-1] is None:
            label_matrices[:, -1] = self.config.mask_fill

        if mask_first_index > 0:
            label_matrices[:mask_first_index, :] = self.config.mask_fill
            label_matrices[:, :mask_first_index] = self.config.mask_fill
        processed_entities_info = []
        for entity_info in entities_info:
            if entity_info['start']==0 and entity_info['end']==0:
                start_token_index, end_token_index = 0, 0
            else:
                start_token_index = self.find_position_in_offsets(entity_info['start'], offsets, sub_word_ids, mask_first_index, offset_length, is_start=True)
                if start_token_index == -1:
                    if self.config.null_to_zero_index:
                        start_token_index, end_token_index = 0, 0
                    else:
                        if self.config.strict:
                            logger.warning(f"cannot find the entity_info : {entity_info}, offsets: {offsets}, we will drop this instance")
                            return None, None
                        logger.warning(f"cannot find the entity_info : {entity_info}, offsets: {offsets}")
                        continue
                else:
                    end_token_index = self.find_position_in_offsets(entity_info['end']-1, offsets, sub_word_ids, start_token_index, offset_length)
                    if self.config.null_to_zero_index and end_token_index == -1:
                        start_token_index, end_token_index = 0, 0
            assert end_token_index != -1, f"entity_info: {entity_info}, offsets: {offsets}"
            label_id = self.vocab.get_index(entity_info['labels'][0])
            label_matrices[start_token_index, end_token_index] = label_id
            entity_info['sub_token_start'] = start_token_index
            entity_info['sub_token_end'] = end_token_index
            processed_entities_info.append(entity_info)

        return label_matrices, processed_entities_info
