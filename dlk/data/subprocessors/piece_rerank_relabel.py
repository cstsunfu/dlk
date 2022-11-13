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

from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
from dlk.utils.logger import Logger
import numpy as np
import pandas as pd
import os

logger = Logger.get_logger()

@subprocessor_config_register('piece_rerank_relabel')
class PieceRerankRelabelConfig(BaseConfig):
    default_config = {
        "_name": "piece_rerank_relabel",
        "config": {
            "train":{
                "input_map": {  
                     "word_ids": "word_ids",
                     "offsets": "offsets",
                     "rank_info": "rank_info",
                },
                "data_set": {                   # for different stage, this processor will process different part of data
                     "train": ['train', 'valid', 'test'],
                     "predict": ['predict'],
                },
                "output_map": {
                     "label_ids": "label_ids",
                },
                "mask_fill": -100,
            },
            "extend_train": "train"
        }
    }
    """Config for PieceRerankRelabel
    Config Example: 
        default_config
    """
    def __init__(self, stage, config: Dict):

        super(PieceRerankRelabelConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.word_ids = self.config['input_map']['word_ids']
        self.offsets = self.config['input_map']['offsets']
        self.mask_fill = self.config['mask_fill']
        self.rank_info = self.config['input_map']['rank_info']
        self.output_labels = self.config['output_map']['label_ids']
        self.post_check(self.config, used=[
            "input_map",
            "data_set",
            "output_map",
            "mask_fill"
        ])


@subprocessor_register('piece_rerank_relabel')
class PieceRerankRelabel(ISubProcessor):
    """
    Relabel the piece rank construct matrix
    """

    def __init__(self, stage: str, config: PieceRerankRelabelConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'piece_rerank_relabel' at stage {self.stage}")
            return

    def process(self, data: Dict)->Dict:
        """PieceRerankRelabel Entry

        Args:
            data: Dict

        Returns: 
            
            relabeled data

        """

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do piece_rerank_relabel on it.')
                continue
            data_set = data['data'][data_set_name]
            if os.environ.get('DISABLE_PANDAS_PARALLEL', 'false') != 'false':
                data_set[self.config.output_labels] = data_set.parallel_apply(self.relabel, axis=1)
            else:
                data_set[self.config.output_labels] = data_set.apply(self.relabel, axis=1)

        return data

    def relabel(self, one_ins: pd.Series):
        """make token label, if use the first piece label please use the 'piece_rerank_firstpiece_relabel'

        Args:
            one_ins: include sentence, rank_info

        Returns: 
            label_matrix for real rank
        """
        word_ids: List = one_ins[self.config.word_ids]
        rank_info = one_ins[self.config.rank_info]
        seq_len = len(word_ids)
        label_matrix = np.full((seq_len, seq_len), 0, dtype=np.int8)
        if not word_ids[0]:
            label_matrix[0] = self.config.mask_fill
            label_matrix[:, 0] = self.config.mask_fill
        if not word_ids[-1]:
            label_matrix[-1] = self.config.mask_fill
            label_matrix[:, -1] = self.config.mask_fill
        # {
        #     "uuid": "uuid",
        #     "pretokenized_words": ['[PAD]', "BCD", "E", "A"],
        #     "rank_info": [0, 3, 1, 2]
        # }
        # ['[CLS]', '[PAD]', 'BC', '##D', 'E', 'A', '[SEP]']
        # [(0, 0), (0, 5), (0, 2), (2, 3), (0, 1), (0, 1), (0, 0)]
        # [None, 0, 1, 1, 2, 3, None]
        # [101, 0, 3823, 2137, 142, 138, 102]
        assert rank_info[0] == 0
        pre_position = None
        for index in rank_info:
            cur_position = word_ids.index(index)
            cur_word_id = word_ids[cur_position]
            if pre_position:
                label_matrix[pre_position][cur_position] = 1
            while cur_position+2 < seq_len and word_ids[cur_position+1] == cur_word_id:
                label_matrix[cur_position][cur_position+1] = 1
                cur_position = cur_position + 1
            pre_position = cur_position
        start_position = word_ids.index(0)
        label_matrix[pre_position][start_position] = 1
        return label_matrix
