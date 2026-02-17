# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import numpy as np
from intc import Base, IntField, ListField, NestField, StrField, cregister

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

    input_map = NestField(value=InputMap, help="input map")

    class OutputMap:
        label_ids = StrField(value="label_ids", help="the label ids")

    output_map = NestField(value=OutputMap, help="output map")
    mask_fill = IntField(value=-100, help="the mask fill value")


@register("subprocessor", "piece_rerank_relabel")
class PieceRerankRelabel(BaseSubProcessor):
    """Relabel the piece rank construct matrix."""

    def __init__(self, stage: str, config: PieceRerankRelabelConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config

    def _relabel_one(self, word_ids, rank_info):
        seq_len = len(word_ids)
        # Note: Datasets/Arrow handles list of lists better than numpy arrays for storage usually,
        # but if the column is a Tensor, numpy is fine.
        # Here we return numpy, which Datasets converts to nested list or Tensor.
        label_matrix = np.full((seq_len, seq_len), 0, dtype=np.int8)

        # Mask special tokens (None word_ids)
        if not word_ids[0]:
            label_matrix[0, :] = self.config.mask_fill
            label_matrix[:, 0] = self.config.mask_fill
        if not word_ids[-1]:
            label_matrix[-1, :] = self.config.mask_fill
            label_matrix[:, -1] = self.config.mask_fill

        # Construct dependency/rank chain
        assert rank_info[0] == 0
        pre_position = None

        for index in rank_info:
            try:
                cur_position = word_ids.index(index)
            except ValueError:
                continue  # Should not happen if data valid

            cur_word_id = word_ids[cur_position]

            if pre_position is not None:
                label_matrix[pre_position][cur_position] = 1

            # Connect all subpieces of the same word
            while (
                cur_position + 1 < seq_len and word_ids[cur_position + 1] == cur_word_id
            ):
                label_matrix[cur_position][cur_position + 1] = 1
                cur_position += 1

            pre_position = cur_position

        # Connect last to first (cycle) or specific root logic
        try:
            start_position = word_ids.index(0)
            if pre_position is not None:
                label_matrix[pre_position][start_position] = 1
        except ValueError:
            pass

        return label_matrix

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:

        w_col = self.config.input_map.word_ids
        r_col = self.config.input_map.rank_info
        out_col = self.config.output_map.label_ids

        if w_col in batch and r_col in batch:
            batch[out_col] = [
                self._relabel_one(w, r) for w, r in zip(batch[w_col], batch[r_col])
            ]

        return batch
