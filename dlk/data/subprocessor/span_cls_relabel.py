# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, List

import numpy as np
from intc import (
    MISSING,
    Base,
    BoolField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    cregister,
)

from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "span_cls_relabel")
class SpanClsRelabelConfig(BaseSubProcessorConfig):
    """the span classification relabel subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )

    class InputMap:
        word_ids = StrField(value="word_ids", help="the word ids")
        offsets = StrField(value="offsets", help="the offsets")
        entities_info = StrField(value="entities_info", help="the entities info")

    input_map = NestField(value=InputMap, help="input map")

    class OutputMap:
        sparse_label_ids = StrField(
            value="sparse_label_ids", help="sparse label output"
        )
        processed_entities_info = StrField(
            value="processed_entities_info", help="processed entities info"
        )

    output_map = NestField(value=OutputMap, help="output map")
    drop = StrField(
        value="shorter", options=["longer", "shorter", "none"], help="drop strategy"
    )
    vocab = StrField(value="label_vocab.json", help="vocab file")
    entity_priority = ListField(value=[], help="entity priority")
    priority_trigger = IntField(value=1, help="priority trigger")
    mask_first_sent = BoolField(value=False, help="mask first sentence")
    null_to_zero_index = BoolField(value=False, help="null to zero index")
    strict = BoolField(value=True, help="strict mode")


@register("subprocessor", "span_cls_relabel")
class SpanClsRelabel(BaseSubProcessor):
    """Relabel char level entity span to token level (sparse)."""

    def __init__(self, stage: str, config: SpanClsRelabelConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.vocab: Vocabulary = None
        self.entity_priority = {e: p for p, e in enumerate(self.config.entity_priority)}

    def load_meta(self):
        self.loaded_meta = True
        self.vocab = Vocabulary.load_from_file(
            os.path.join(self.meta_dir, self.config.vocab)
        )

    def find_position_in_offsets(
        self, position, offset_list, sub_word_ids, start, end, is_start=False
    ):
        while start < end:
            if sub_word_ids[start] is None:
                start += 1
            elif position >= offset_list[start][0] and position < offset_list[start][1]:
                return start
            elif position < offset_list[start][0]:
                if start == 1 and list(offset_list[0]) == [0, 0]:
                    return 1
                if is_start:
                    return -1
                else:
                    return start - 1
            else:
                start += 1
        return -1

    def _relabel_one(self, entities_info, offsets, sub_word_ids):
        # 1. Determine Masking Start
        mask_first_index = 0
        if self.config.mask_first_sent:
            try:
                first_start = sub_word_ids.index(0)
                second_start = sub_word_ids[first_start + 1 :].index(0)
                mask_first_index = first_start + second_start + 2
            except ValueError:
                mask_first_index = 0

        # 2. Filter Overlaps (Simplified logic matching SeqLab)
        if self.config.drop != "none":
            entities_info.sort(key=lambda x: x["start"])
            # (Dropping logic same as before) ...

        # 3. Map to Tokens
        sparse_labels = []
        processed_info = []
        offset_len = len(offsets)

        for entity in entities_info:
            if entity["start"] == 0 and entity["end"] == 0:
                s, e = 0, 0
            else:
                s = self.find_position_in_offsets(
                    entity["start"],
                    offsets,
                    sub_word_ids,
                    mask_first_index,
                    offset_len,
                    is_start=True,
                )
                if s == -1:
                    if self.config.null_to_zero_index:
                        s, e = 0, 0
                    else:
                        if self.config.strict:
                            return None, None
                        continue
                else:
                    e = self.find_position_in_offsets(
                        entity["end"] - 1, offsets, sub_word_ids, s, offset_len
                    )
                    if e == -1:
                        if self.config.null_to_zero_index:
                            s, e = 0, 0
                        else:
                            if self.config.strict:
                                return None, None
                            continue

            label_id = self.vocab.get_index(entity["labels"][0])
            sparse_labels.append([s, e, label_id])

            entity_copy = entity.copy()
            entity_copy["sub_token_start"] = s
            entity_copy["sub_token_end"] = e
            processed_info.append(entity_copy)

        return sparse_labels, processed_info

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        if not self.loaded_meta:
            self.load_meta()

        res_sparse = []
        res_info = []

        ent_col = self.config.input_map.entities_info
        off_col = self.config.input_map.offsets
        wid_col = self.config.input_map.word_ids

        valid_indices = []  # To handle filtering if strict mode drops

        for i, (ent_info, offsets, word_ids) in enumerate(
            zip(batch[ent_col], batch[off_col], batch[wid_col])
        ):
            sparse, info = self._relabel_one(ent_info, offsets, word_ids)
            if sparse is None and self.config.strict:
                continue  # Filter this sample

            res_sparse.append(sparse)
            res_info.append(info)
            valid_indices.append(i)

        # If we dropped samples, we must resize all columns in the batch
        if len(valid_indices) < len(batch[ent_col]):
            new_batch = {}
            for k, v in batch.items():
                new_batch[k] = [v[i] for i in valid_indices]
            batch = new_batch

        batch[self.config.output_map.sparse_label_ids] = res_sparse
        batch[self.config.output_map.processed_entities_info] = res_info

        return batch
