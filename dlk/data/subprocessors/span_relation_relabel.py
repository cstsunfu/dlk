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
            "train":{
                "input_map": { 
                     "word_ids": "word_ids",
                     "offsets": "offsets",
                     "relations_info": "relations_info",
                     "entities_index_info": "entities_index_info",
                },
                "data_set": {
                     "train": ['train', 'valid', 'test'],
                },
                "output_map": {
                    "relation_label_ids": "relation_label_ids",
                },
                "drop": "none", # 'longer'/'shorter'/'none', if entities is overlap, will remove by rule
                "vocab": "label_vocab#relation", # usually provided by the "token_gather" module
                "pad": -100,
            },
        }
    }
    f"""Config for SpanRelationRelabel
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
        self.pad = self.config['pad']
        self.relations_info = self.config['input_map']['relations_info']
        self.entities_index_info = self.config['input_map']['entities_index_info']
        self.vocab = self.config['vocab']
        self.output_labels = self.config['output_map']['relation_label_ids']
        self.post_check(self.config, used=[
            "drop",
            "vocab",
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

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do span_relation_relabel on it.')
                continue
            data_set = data['data'][data_set_name]
            if os.environ.get('DISABLE_PANDAS_PARALLEL', 'false') != 'false':
                data_set[[self.config.output_labels,
                ]] = data_set.parallel_apply(self.relabel, axis=1, result_type='expand')
            else:
                data_set[[self.config.output_labels,
                ]] = data_set.apply(self.relabel, axis=1, result_type='expand')

        return data

    def relabel(self, one_ins: pd.Series):
        """make token label, if use the first piece label please use the 'span_cls_firstpiece_relabel'

        Args:
            one_ins: include sentence, entity_info, offsets

        Returns: 
            labels(labels for each subtoken)

        """
        relations_info = one_ins[self.config.relations_info]
        sub_word_ids = one_ins[self.config.word_ids]
        offsets = one_ins[self.config.offsets]
        cur_token_index = 0
        offset_length = len(offsets)

        unk_id = self.vocab.get_index(self.vocab.unknown)
        mask_matrices = np.full((offset_length, offset_length), self.config.pad)
        mask_matrices = np.tril(mask_matrices, k=-1)

        unk_matrices = np.full((offset_length, offset_length), unk_id)
        unk_matrices = np.triu(unk_matrices, k=0)

        label_matrices = unk_matrices + mask_matrices
        if sub_word_ids[0] is None:
            label_matrices[0, 0] = self.config.pad
        if sub_word_ids[-1] is None:
            label_matrices[-1, -1] = self.config.pad

        for entity_info in entities_info:
            start_token_index = self.find_position_in_offsets(entity_info['start'], offsets, sub_word_ids, cur_token_index, offset_length, is_start=True)
            if start_token_index == -1:
                logger.warning(f"cannot find the entity_info : {entity_info}, offsets: {offsets} ")
                continue
            end_token_index = self.find_position_in_offsets(entity_info['end']-1, offsets, sub_word_ids, start_token_index, offset_length)
            assert end_token_index != -1, f"entity_info: {entity_info}, offsets: {offsets}"
            label_id = self.vocab.get_index(entity_info['labels'][0])
            label_matrices[start_token_index, end_token_index] = label_id

        if not self.config.clean_droped_entity:
            entities_info = one_ins[self.config.entities_info]
        return label_matrices, entities_info
