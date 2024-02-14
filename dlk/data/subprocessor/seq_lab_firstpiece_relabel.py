# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, List, Set

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


@cregister("subprocessor", "seq_lab_firstpiece_relabel")
class SeqLabFirstPieceRelabelConfig(BaseSubProcessorConfig):
    """the sequence labeling firstpiece relabel subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict_data_set = ListField(
        value=["predict"],
        suggestions=[["predict"]],
        help="the data set should be processed for predict stage",
    )
    online_data_set = ListField(
        value=["online"],
        suggestions=[["online"]],
        help="the data set should be processed for online stage",
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
        labels = StrField(value="labels", help="the label names")
        gather_index = StrField(value="gather_index", help="the gather index")
        word_word_ids = StrField(value="word_ids", help="the word word ids")
        word_offsets = StrField(value="offsets", help="the word offsets")

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )
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
        help="the entity priority, when conflict, will keep the entity with the highest priority",
    )
    priority_trigger = IntField(
        value=1,
        help="if the overlap entity abs(length_a - length_b)<=priority_trigger, will trigger the entity_priority strategy",
    )


@register("subprocessor", "seq_lab_firstpiece_relabel")
class SeqLabFirstPieceRelabel(BaseSubProcessor):
    """Relabel the json data to bio.

    And gather the firstpiece subword index to deliver $gather_index; for get the firstpiece index we must use the $word_ids(generator by tokenizer)
    """

    def __init__(
        self, stage: str, config: SeqLabFirstPieceRelabelConfig, meta_dir: str
    ):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config
        self.entity_priority = {
            entity: priority
            for priority, entity in enumerate(self.config.entity_priority)
        }

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

        if self.stage in {"predict", "online"}:
            data[
                [
                    self.config.output_map.gather_index,
                    self.config.output_map.word_word_ids,
                    self.config.output_map.word_offsets,
                ]
            ] = data.apply(self.relabel, axis=1, result_type="expand")
        else:
            data[
                [
                    self.config.output_map.labels,
                    self.config.output_map.gather_index,
                    self.config.output_map.word_word_ids,
                    self.config.output_map.word_offsets,
                    self.config.input_map.entities_info,
                ]
            ] = data.apply(self.relabel, axis=1, result_type="expand")
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
        """make word label, first merge the token_offset to word_offsets then generate word label from  word_offsets

        Args:
            one_ins: include sentence, entity_info, offsets

        Returns:
            For train stage:
                labels(labels for each word not subtoken), gather_index(real token index of labels), word_ids(sub_word_ids, updated to whole word), word_offsets(token_offsets updated to whole word), entities_info
            For predict / online stage:
                gather_index(real token index of labels), word_ids(sub_word_ids, updated to whole word), word_offsets(token_offsets updated to whole word)
        """
        pre_clean_entities_info = one_ins[self.config.input_map.entities_info]
        pre_clean_entities_info.sort(key=lambda x: x["start"])
        offsets = one_ins[self.config.input_map.offsets]
        sub_word_ids = one_ins[self.config.input_map.word_ids]
        if not sub_word_ids:
            logger.warning(
                f"entity_info: {pre_clean_entities_info}, offsets: {offsets} "
            )

        gather_index = []
        pre_word_id = -1
        word_offset = []
        word_offsets = []
        word_ids = []
        for i, (token_offset, word_id) in enumerate(zip(offsets, sub_word_ids)):
            if word_id != pre_word_id:
                gather_index.append(i)
                word_ids.append(word_id)
                if word_offset:
                    word_offsets.append(word_offset)
                word_offset = list(token_offset)
                pre_word_id = word_id
            else:
                assert word_offset
                word_offset[1] = token_offset[1]
        if word_offset:
            word_offsets.append(word_offset)
        if self.stage in {"predict", "online"}:
            return gather_index, word_ids, word_offsets

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

        cur_token_index = 0
        offset_length = len(word_offsets)
        sub_labels = []
        for entity_info in entities_info:
            start_token_index = self.find_position_in_offsets(
                entity_info["start"],
                word_offsets,
                word_ids,
                cur_token_index,
                offset_length,
                is_start=True,
            )
            if start_token_index == -1:
                logger.warning(
                    f"cannot find the entity_info : {entity_info}, word_offsets: {word_offsets} "
                )
                continue
            for _ in range(start_token_index - cur_token_index):
                sub_labels.append("O")
            end_token_index = self.find_position_in_offsets(
                entity_info["end"] - 1,
                word_offsets,
                word_ids,
                start_token_index,
                offset_length,
            )
            assert (
                end_token_index != -1
            ), f"entity_info: {entity_info}, word_offsets: {word_offsets}"
            sub_labels.append("B-" + entity_info["labels"][0])
            for _ in range(end_token_index - start_token_index):
                sub_labels.append("I-" + entity_info["labels"][0])
            cur_token_index = end_token_index + 1
        assert cur_token_index <= offset_length
        for _ in range(offset_length - cur_token_index):
            sub_labels.append("O")

        if word_ids[0] is None:
            sub_labels[0] = self.config.start_label

        if word_ids[offset_length - 1] is None:
            sub_labels[offset_length - 1] = self.config.end_label

        if len(sub_labels) != len(gather_index):
            logger.error(f"{len(sub_labels)} vs {len(gather_index)}")
            for i in one_ins:
                logger.error(f"{i}")
            raise PermissionError

        if not self.config.clean_droped_entity:
            entities_info = one_ins[self.config.input_map.entities_info]
        return sub_labels, gather_index, word_ids, word_offsets, entities_info
