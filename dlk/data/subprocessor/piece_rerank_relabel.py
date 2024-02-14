# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, List, Set

import numpy as np
import pandas as pd
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)

from dlk.utils.io import open
from dlk.utils.register import register

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "piece_rerank_relabel")
class PieceRerankRelabelConfig(BaseSubProcessorConfig):
    """the piece rerank relabel subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"], []],
        help="the data set should be processed for train stage",
    )

    class InputMap:
        word_ids = StrField(value="word_ids", help="the word ids")
        offsets = StrField(value="offsets", help="the offsets")
        rank_info = StrField(value="rank_info", help="the rank info")

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        label_ids = StrField(value="label_ids", help="the label ids")

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )
    mask_fill = IntField(value=-100, help="the mask fill value")


@register("subprocessor", "piece_rerank_relabel")
class PieceRerankRelabel(BaseSubProcessor):
    """
    Relabel the piece rank construct matrix
    """

    def __init__(self, stage: str, config: PieceRerankRelabelConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """Character gather entry

        Args:
            data: one is like
            >>> {
            >>>     "uuid": "uuid",
            >>>     "pretokenized_words": ['[PAD]', "BCD", "E", "A"],
            >>>     "rank_info": [0, 3, 1, 2]
            >>> }

            deliver_meta:
                ignore
        Returns:
            processed data
        """
        data[self.config.output_map.label_ids] = data.apply(self.relabel, axis=1)

        return data

    def relabel(self, one_ins: pd.Series):
        """make token label, if use the first piece label please use the 'piece_rerank_firstpiece_relabel'

        Args:
            one_ins: include sentence, rank_info

        Returns:
            label_matrix for real rank
        """
        word_ids: List = one_ins[self.config.input_map.word_ids]
        rank_info = one_ins[self.config.input_map.rank_info]
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
            while (
                cur_position + 2 < seq_len and word_ids[cur_position + 1] == cur_word_id
            ):
                label_matrix[cur_position][cur_position + 1] = 1
                cur_position = cur_position + 1
            pre_position = cur_position
        start_position = word_ids.index(0)
        label_matrix[pre_position][start_position] = 1
        return label_matrix
