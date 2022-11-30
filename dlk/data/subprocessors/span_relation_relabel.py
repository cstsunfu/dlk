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
import json

logger = Logger.get_logger()

@subprocessor_config_register('span_relation_relabel')
class SpanRelationRelabelConfig(BaseConfig):
    default_config = {
        "_name": "span_relation_relabel",
        "config": {
            "train": {
                "input_map": { 
                     "word_ids": "word_ids",
                     "offsets": "offsets",
                     "relations_info": "relations_info",
                     "processed_entities_info": "processed_entities_info",
                },
                "data_set": {
                     "train": ['train', 'valid', 'test'],
                },
                "output_map": {
                    "label_ids": "relation_label_ids",
                },
                "drop": "none", # 'longer'/'shorter'/'none', if entities is overlap, will remove by rule
                "vocab": "label_vocab#relation", # usually provided by the "token_gather" module
                "mask_fill": -100,
                "label_seperate": False, # seperate differences types of relations to different label matrix or use universal matrix(universal matrix may have conflict issue, like the 2 entities of the head of 2 different relations is same)
                "sym": True, # whther the from entity and end entity can swap in relations( if sym==True, we can just calc the upper trim and set down trim as -100 to ignore)
                "strict": True, # if strict == True, will drop the unvalid sample
            },
            "extend_train": "train"
        }
    }
    f"""Config for SpanRelationRelabel
    Config Example:
        default_config
    """
    # {json.dumps(default_config, indent=4, ensure_ascii=False)}
    def __init__(self, stage, config: Dict):

        super(SpanRelationRelabelConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.offsets = self.config['input_map']['offsets']
        self.word_ids = self.config['input_map']['word_ids']
        self.mask_fill = self.config['mask_fill']
        self.sym = self.config['sym']
        self.strict = self.config['strict']
        self.label_seperate = self.config['label_seperate']
        self.relations_info = self.config['input_map']['relations_info']
        self.processed_entities_info = self.config['input_map']['processed_entities_info']
        self.vocab = self.config['vocab']
        self.output_labels = self.config['output_map']['label_ids']
        self.post_check(self.config, used=[
            "drop",
            "vocab",
            "mask_fill",
            "label_seperate",
            "sym",
            "input_map",
            "data_set",
            "output_map",
        ])


@subprocessor_register('span_relation_relabel')
class SpanRelationRelabel(ISubProcessor):
    """
    Relabel the json data to bio
    """

    def __init__(self, stage: str, config: SpanRelationRelabelConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        self.vocab = None
        if not self.data_set:
            logger.info(f"Skip 'span_relation_relabel' at stage {self.stage}")
            return

    def process(self, data: Dict)->Dict:
        """SpanRelationRelabel Entry

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
                logger.info(f'The {data_set_name} not in data. We will skip do span_relation_relabel on it.')
                continue
            data_set = data['data'][data_set_name]
            if os.environ.get('DISABLE_PANDAS_PARALLEL', 'false') != 'false':
                data_set[self.config.output_labels] = data_set.parallel_apply(self.relabel, axis=1)
            else:
                data_set[self.config.output_labels] = data_set.apply(self.relabel, axis=1)
            if self.config.strict:
                data_set.dropna(axis=0, inplace=True)
                data_set.reset_index(inplace=True)

        return data

    def relabel(self, one_ins: pd.Series):
        """make token label, if use the first piece label please use the 'span_cls_firstpiece_relabel'

        Args:
            one_ins: include sentence, entity_info, offsets

        Returns: 
            labels(labels for each subtoken)

        """
        relations_info = one_ins[self.config.relations_info]
        processed_entities_info = one_ins[self.config.processed_entities_info]
        entities_id_info_map = {}
        for processed_entity_info in processed_entities_info:
            entities_id_info_map[processed_entity_info['entity_id']] = processed_entity_info
        sub_word_ids = one_ins[self.config.word_ids]
        offsets = one_ins[self.config.offsets]
        offset_length = len(offsets)

        unknown_id = self.vocab.get_index(self.vocab.unknown)
        unknown_matrix = np.full((offset_length, offset_length), unknown_id)
        if self.config.sym:
            mask_matrix = np.full((offset_length, offset_length), self.config.mask_fill)
            mask_matrix = np.tril(mask_matrix, k=-1)
            unknown_matrix = np.triu(unknown_matrix, k=0)
            label_matrix = unknown_matrix + mask_matrix
        else:
            label_matrix = unknown_matrix
        if sub_word_ids[0] is None:
            label_matrix[0, :] = self.config.mask_fill
        if sub_word_ids[-1] is None:
            label_matrix[:, -1] = self.config.mask_fill
        if self.config.label_seperate:
            label_matrices = self.get_seperate_label_matrix(label_matrix, relations_info, entities_id_info_map)
        else:
            label_matrices = self.get_universal_label_matrix(label_matrix, relations_info, entities_id_info_map)

        return label_matrices

    def _get_entities_index(self, relation_info, entities_id_info_map):
        """get the from entity head, to entity head, from entity tail, to entity tail
        """
        from_index = relation_info['from']
        to_index = relation_info['to']
        if from_index not in entities_id_info_map or to_index not in entities_id_info_map:
            return None, None, None, None
        from_entity = entities_id_info_map[from_index]
        from_start_index = from_entity['sub_token_start']
        from_end_index = from_entity['sub_token_end']
        to_entity = entities_id_info_map[to_index]
        to_start_index = to_entity['sub_token_start']
        to_end_index = to_entity['sub_token_end']

        if self.config.sym:
            if from_start_index > to_start_index:
                from_start_index, to_start_index = to_start_index, from_start_index
            if from_end_index > to_end_index:
                from_end_index, to_end_index = to_end_index, from_end_index
        return from_start_index, to_start_index, from_end_index, to_end_index 

    def get_seperate_label_matrix(self, label_matrix, relations_info: List, entities_id_info_map: Dict[int, Dict]):
        """seperate different type of relations to different matrix(as one dimension)

        Args:
            label_matrices: matrix
            relations_info: like
                [
                    {
                        "labels": [
                            "belong_to"
                        ],
                        "from": 1, # index of entity
                        "to": 0,
                    }
                ]
            entities_id_info_map: generate by span_cls_relabel

        Returns: 
            label_id matrix

        """
        label_types = len(self.vocab) - 1
        label_matrix = np.expand_dims(label_matrix, 0).repeat(2*label_types, 0)

        for relation_info in relations_info:
            label_id = self.vocab.word2idx[relation_info['labels'][0]]
            from_start_index, to_start_index, from_end_index, to_end_index = self._get_entities_index(relation_info, entities_id_info_map)
            if from_start_index is None:
                if self.config.strict:
                    logger.warning(f"Cannot get the relation of {json.dumps(relation_info)}, we will drop this instance")
                    return None
                continue
            # NOTE: label_id ==0 is unknown
            label_matrix[(label_id-1)*2][from_start_index][to_start_index] = 1
            label_matrix[(label_id-1)*2+1][from_end_index][to_end_index] = 1
        return label_matrix

    def get_universal_label_matrix(self, label_matrix, relations_info: List, entities_id_info_map: Dict[int, Dict]):
        """different type of relations to one matrix

        Args:
            label_matrices: matrix
            relations_info: like
                [
                    {
                        "labels": [
                            "belong_to"
                        ],
                        "from": 1, # index of entity
                        "to": 0,
                    }
                ]
            entities_id_info_map: generate by span_cls_relabel

        Returns:
            label_id matrix
        """
        label_matrix = np.expand_dims(label_matrix, 0).repeat(2, 0)
        for relation_info in relations_info:
            label_id = self.vocab.word2idx[relation_info['labels'][0]]
            from_start_index, to_start_index, from_end_index, to_end_index = self._get_entities_index(relation_info, entities_id_info_map)
            if from_start_index is None:
                if self.config.strict:
                    logger.warning(f"Cannot get the relation of {json.dumps(relation_info)}, we will drop this instance")
                    return None
                continue
            label_matrix[0][from_start_index][to_start_index] = label_id
            label_matrix[1][from_end_index][to_end_index] = label_id
        return label_matrix
