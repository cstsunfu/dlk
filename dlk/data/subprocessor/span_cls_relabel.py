# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
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

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        label_ids = StrField(value="label_ids", help="the label ids")
        processed_entities_info = StrField(
            value="processed_entities_info", help="the processed entities info"
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )
    drop = StrField(
        value="shorter",
        options=["longer", "shorter", "none"],
        help="the drop strategy for the overlap entities",
    )
    vocab = StrField(value="label_vocab.json", help="the vocab for the label")

    entity_priority = ListField(
        value=[],
        suggestions=[["Product", "Brand"]],
        help="the entity priority, when conflict, will keep the entity with the highest priority",
    )
    priority_trigger = IntField(
        value=1,
        help="if the overlap entity abs(length_a - length_b)<=priority_trigger, will trigger the entity_priority strategy",
    )
    mask_fill = IntField(value=-100, help="the mask fill for the label")
    mask_first_sent = BoolField(
        value=False, help="whether mask the first sentence for anwering"
    )
    null_to_zero_index = BoolField(
        value=False,
        help="if cannot find the entity, set to point to the first(zero) index token",
    )
    strict = BoolField(
        value=True, help="if strict == True, will drop the invalid sample"
    )


@register("subprocessor", "span_cls_relabel")
class SpanClsRelabel(BaseSubProcessor):
    """
    Relabel the char level entity span to token level and construct matrix
    """

    def __init__(self, stage: str, config: SpanClsRelabelConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config
        self.vocab: Vocabulary = None
        self.entity_priority = {
            entity: priority
            for priority, entity in enumerate(self.config.entity_priority)
        }

    def load_meta(self):
        self.loaded_meta = True
        self.vocab = Vocabulary.load_from_file(
            os.path.join(self.meta_dir, self.config.vocab)
        )
        assert (
            self.vocab.word2idx[self.vocab.unknown] == 0
        ), f"For span_cls_relabel, 'unknown' must be index 0, and other labels as 1...num_label"
        assert (
            not self.vocab.pad
        ), f"For span_cls_relabel, 'pad' must be index 0, and other labels as 1...num_label"

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """firstpiece relabel the data

        Args:
            data: one is like

            >>> {
            >>>     "uuid": '**-**-**-**'
            >>>     "sentence": "Mie Merah - Buah Bit",
            >>>     "offsets": see the offsets in fast_tokenizer
            >>>     "entities_info": [
            >>>                 {
            >>>                     "end": 9,
            >>>                     "start": 0,
            >>>                     "labels": [
            >>>                         "Product"
            >>>                     ]
            >>>                 },
            >>>             ]
            >>>         },
            >>>     ],
            >>> },

            deliver_meta:
                ignore
        Returns:
            relabeld data
        """
        if not self.loaded_meta:
            self.load_meta()

        data[
            [
                self.config.output_map.label_ids,
                self.config.output_map.processed_entities_info,
            ]
        ] = data.apply(self.relabel, axis=1, result_type="expand")
        if self.config.strict:
            data.dropna(axis=0, inplace=True)
            data.reset_index(inplace=True)

        return data

    def find_position_in_offsets(
        self,
        position: int,
        offset_list: List,
        sub_word_ids: List,
        start: int,
        end: int,
        is_start: bool = False,
    ):
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
        while start < end:
            if sub_word_ids[start] is None:
                start += 1
            elif position >= offset_list[start][0] and position < offset_list[start][1]:
                return start
            elif position < offset_list[start][0]:
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
        pre_clean_entities_info: List = one_ins[self.config.input_map.entities_info]
        if self.config.drop != "none":
            pre_clean_entities_info.sort(key=lambda x: x["start"])
        offsets: List = one_ins[self.config.input_map.offsets]
        sub_word_ids: List = one_ins[self.config.input_map.word_ids]

        if self.config.mask_first_sent:
            first_start = sub_word_ids.index(0)
            second_start = sub_word_ids[first_start + 1 :].index(0)
            mask_first_index = first_start + second_start + 2
        else:
            mask_first_index = 0  # if there is only one sentence, set to 0

        entities_info = []
        pre_end = -1
        pre_length = 0
        pre_label = ""
        for entity_info in pre_clean_entities_info:
            assert (
                len(entity_info["labels"]) == 1
            ), f"currently we just support one label for one entity"
            if entity_info["start"] < pre_end:  # if overlap will remove one
                if self.config.drop == "none":
                    pass
                elif (
                    abs(entity_info["end"] - entity_info["start"] - pre_length)
                    <= self.config.priority_trigger
                ):
                    pre_label_order = self.entity_priority.get(pre_label, 1e9)
                    label_order = self.entity_priority.get(
                        entity_info["labels"][0], 1e9
                    )
                    if label_order < pre_label_order:
                        entities_info.pop()
                    else:
                        continue
                elif self.config.drop == "shorter":
                    if entity_info["end"] - entity_info["start"] > pre_length:
                        entities_info.pop()
                    else:
                        continue
                elif self.config.drop == "longer":
                    if entity_info["end"] - entity_info["start"] < pre_length:
                        entities_info.pop()
                    else:
                        continue
                else:
                    raise PermissionError(
                        f"The drop method must in 'none'/'shorter'/'longer'"
                    )
            pre_label = entity_info["labels"][0]
            entities_info.append(entity_info)
            pre_end = entity_info["end"]
            pre_length = entity_info["end"] - entity_info["start"]

        offset_length = len(offsets)

        unknown_id = self.vocab.get_index(self.vocab.unknown)
        mask_matrices = np.full(
            (offset_length, offset_length), self.config.mask_fill, dtype=np.int8
        )
        mask_matrices = np.tril(mask_matrices, k=-1)

        unknown_matrices = np.full(
            (offset_length, offset_length), unknown_id, dtype=np.int8
        )
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
            if entity_info["start"] == 0 and entity_info["end"] == 0:
                start_token_index, end_token_index = 0, 0
            else:
                start_token_index = self.find_position_in_offsets(
                    entity_info["start"],
                    offsets,
                    sub_word_ids,
                    mask_first_index,
                    offset_length,
                    is_start=True,
                )
                if start_token_index == -1:
                    if self.config.null_to_zero_index:
                        start_token_index, end_token_index = 0, 0
                    else:
                        if self.config.strict:
                            logger.warning(
                                f"cannot find the entity_info : {entity_info}, offsets: {offsets}, we will drop this instance"
                            )
                            return None, None
                        logger.warning(
                            f"cannot find the entity_info : {entity_info}, offsets: {offsets}"
                        )
                        continue
                else:
                    end_token_index = self.find_position_in_offsets(
                        entity_info["end"] - 1,
                        offsets,
                        sub_word_ids,
                        start_token_index,
                        offset_length,
                    )
                    if self.config.null_to_zero_index and end_token_index == -1:
                        start_token_index, end_token_index = 0, 0
            assert (
                end_token_index != -1
            ), f"entity_info: {entity_info}, offsets: {offsets}"
            label_id = self.vocab.get_index(entity_info["labels"][0])
            label_matrices[start_token_index, end_token_index] = label_id
            entity_info["sub_token_start"] = start_token_index
            entity_info["sub_token_end"] = end_token_index
            processed_entities_info.append(entity_info)

        return label_matrices, processed_entities_info
