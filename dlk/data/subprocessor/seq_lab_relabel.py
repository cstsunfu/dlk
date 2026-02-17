# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple

from intc import Base, BoolField, IntField, ListField, NestField, StrField, cregister

from dlk.utils.register import register

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "seq_lab_relabel")
class SeqLabRelabelConfig(BaseSubProcessorConfig):
    """the sequence labeling relabel subprocessor"""

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
        labels = StrField(value="labels", help="the label names")

    output_map = NestField(value=OutputMap, help="output map")

    drop = StrField(
        value="shorter",
        options=["longer", "shorter", "none"],
        help="the drop strategy for the overlap entities",
    )
    start_label = StrField(value="S", help="the start label")
    end_label = StrField(value="E", help="the end label")
    clean_droped_entity = BoolField(
        value=True, help="whether clean the dropped entity for calc metrics"
    )
    entity_priority = ListField(
        value=[],
        suggestions=[["Product", "Brand"]],
        help="the entity priority",
    )
    priority_trigger = IntField(
        value=1,
        help="priority trigger threshold",
    )


@register("subprocessor", "seq_lab_relabel")
class SeqLabRelabel(BaseSubProcessor):
    """Relabel the json data to bio for Sequence Labeling."""

    def __init__(self, stage: str, config: SeqLabRelabelConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.entity_priority = {
            entity: priority
            for priority, entity in enumerate(self.config.entity_priority)
        }

    def find_position_in_offsets(
        self,
        position: int,
        offset_list: List[Tuple[int, int]],
        sub_word_ids: List[int],
        start: int,
        end: int,
        is_start: bool = False,
    ) -> int:
        """Find token index covering the position."""
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

    def _relabel_one(
        self, entities_info: List[Dict], offsets: List, sub_word_ids: List
    ):
        """Logic to generate BIO labels for a single sample."""
        # 1. Resolve Overlaps
        clean_entities = []
        if self.config.drop != "none" or self.config.entity_priority:
            entities_info.sort(key=lambda x: x["start"])
            pre_end = -1
            pre_length = 0
            pre_label = ""

            for entity in entities_info:
                # Logic for overlap dropping
                if entity["start"] < pre_end:
                    drop_current = False
                    drop_prev = False

                    # Priority Check
                    if (
                        abs(entity["end"] - entity["start"] - pre_length)
                        <= self.config.priority_trigger
                    ):
                        pre_p = self.entity_priority.get(pre_label, 1e9)
                        cur_p = self.entity_priority.get(entity["labels"][0], 1e9)
                        if cur_p < pre_p:
                            drop_prev = True
                        else:
                            drop_current = True
                    # Length Check
                    elif self.config.drop == "shorter":
                        if entity["end"] - entity["start"] > pre_length:
                            drop_prev = True
                        else:
                            drop_current = True
                    elif self.config.drop == "longer":
                        if entity["end"] - entity["start"] < pre_length:
                            drop_prev = True
                        else:
                            drop_current = True

                    if drop_prev:
                        clean_entities.pop()
                    elif drop_current:
                        continue

                clean_entities.append(entity)
                pre_end = entity["end"]
                pre_length = entity["end"] - entity["start"]
                pre_label = entity["labels"][0]
        else:
            clean_entities = entities_info

        # 2. Generate Labels
        offset_len = len(offsets)
        sub_labels = ["O"] * offset_len
        cur_token_idx = 0

        for entity in clean_entities:
            label = entity["labels"][0]
            start_idx = self.find_position_in_offsets(
                entity["start"],
                offsets,
                sub_word_ids,
                cur_token_idx,
                offset_len,
                is_start=True,
            )

            if start_idx == -1:
                continue

            end_idx = self.find_position_in_offsets(
                entity["end"] - 1, offsets, sub_word_ids, start_idx, offset_len
            )

            if end_idx != -1:
                sub_labels[start_idx] = f"B-{label}"
                for i in range(start_idx + 1, end_idx + 1):
                    sub_labels[i] = f"I-{label}"
                cur_token_idx = end_idx + 1

        # 3. Handle Special Tokens (CLS/SEP)
        if sub_word_ids[0] is None:
            sub_labels[0] = self.config.start_label
        if sub_word_ids[-1] is None:
            sub_labels[-1] = self.config.end_label

        return sub_labels, clean_entities

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """Process batch to relabel sequences."""

        ent_col = self.config.input_map.entities_info
        off_col = self.config.input_map.offsets
        word_col = self.config.input_map.word_ids

        out_lab_col = self.config.output_map.labels

        if ent_col in batch:
            res_labels = []
            res_entities = []

            for ent_info, offsets, word_ids in zip(
                batch[ent_col], batch[off_col], batch[word_col]
            ):
                labels, final_entities = self._relabel_one(ent_info, offsets, word_ids)
                res_labels.append(labels)
                res_entities.append(final_entities)

            batch[out_lab_col] = res_labels
            if self.config.clean_droped_entity:
                batch[ent_col] = res_entities

        return batch
