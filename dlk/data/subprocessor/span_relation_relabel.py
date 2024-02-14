# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
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

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "span_relation_relabel")
class SpanRelationRelabelConfig(BaseSubProcessorConfig):
    """the span relation relabel subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )

    class InputMap:
        word_ids = StrField(value="word_ids", help="the word ids")
        offsets = StrField(value="offsets", help="the offsets")
        relations_info = StrField(value="relations_info", help="the relations info")
        processed_entities_info = StrField(
            value="processed_entities_info", help="the processed entities info"
        )

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
    vocab = StrField(
        value="label_vocab#relation.json", help="the vocab for the relation label"
    )

    mask_fill = IntField(value=-100, help="the mask fill for the label")
    label_seperate = BoolField(
        value=False,
        help="separate differences types of relations to different label matrix or use universal matrix(universal matrix may have conflict issue, like the 2 entities of the head of 2 different relations is same)",
    )
    sym = BoolField(
        value=True,
        help="whether the from entity and end entity can swap in relations( if sym==True, we can just calc the upper trim and set down trim as -100 to ignore)",
    )
    strict = BoolField(
        value=True, help="if strict == True, will drop the invalid sample"
    )


@register("subprocessor", "span_relation_relabel")
class SpanRelationRelabel(BaseSubProcessor):
    """
    Relabel the json data to bio
    """

    def __init__(self, stage: str, config: SpanRelationRelabelConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config
        self.vocab: Vocabulary = None

    def load_meta(self):
        self.loaded_meta = True
        self.vocab = Vocabulary.load_from_file(
            os.path.join(self.meta_dir, self.config.vocab)
        )
        assert (
            self.vocab.word2idx[self.vocab.unknown] == 0
        ), f"For span_relation_relabel, 'unknown' must be index 0, and other labels as 1...num_label"
        assert (
            not self.vocab.pad
        ), f"For span_relation_relabel, 'pad' must be index 0, and other labels as 1...num_label"

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """firstpiece relabel the data

        Args:
            data: one is like

            >>> {
            >>>     "uuid": '**-**-**-**'
            >>>     "sentence": "Mie Merah - Buah Bit",
            >>>     "offsets": see the offsets in fast_tokenizer
            >>>     "processed_entities_info": generate by `span_cls_relabel`
            >>>     "relations_info": like
            >>>         [
            >>>             {
            >>>                 "labels": [
            >>>                     "belong_to"
            >>>                 ],
            >>>                 "from": 1, # index of entity
            >>>                 "to": 0,
            >>>             }
            >>>         ]
            >>> },

            deliver_meta:
                ignore
        Returns:
            relabeld data
        """
        if not self.loaded_meta:
            self.load_meta()

        data[self.config.output_map.label_ids] = data.apply(self.relabel, axis=1)
        if self.config.strict:
            data.dropna(axis=0, inplace=True)
            data.reset_index(inplace=True)

        return data

    def relabel(self, one_ins: pd.Series):
        """make token label, if use the first piece label please use the 'span_cls_firstpiece_relabel'

        Args:
            one_ins: include sentence, entity_info, offsets

        Returns:
            labels(labels for each subtoken)

        """
        relations_info = one_ins[self.config.input_map.relations_info]
        processed_entities_info = one_ins[self.config.input_map.processed_entities_info]
        entities_id_info_map = {}
        for processed_entity_info in processed_entities_info:
            entities_id_info_map[
                processed_entity_info["entity_id"]
            ] = processed_entity_info
        sub_word_ids = one_ins[self.config.input_map.word_ids]
        offsets = one_ins[self.config.input_map.offsets]
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
            label_matrices = self.get_seperate_label_matrix(
                label_matrix, relations_info, entities_id_info_map
            )
        else:
            label_matrices = self.get_universal_label_matrix(
                label_matrix, relations_info, entities_id_info_map
            )

        return label_matrices

    def _get_entities_index(self, relation_info, entities_id_info_map):
        """get the from entity head, to entity head, from entity tail, to entity tail"""
        from_index = relation_info["from"]
        to_index = relation_info["to"]
        if (
            from_index not in entities_id_info_map
            or to_index not in entities_id_info_map
        ):
            return None, None, None, None
        from_entity = entities_id_info_map[from_index]
        from_start_index = from_entity["sub_token_start"]
        from_end_index = from_entity["sub_token_end"]
        to_entity = entities_id_info_map[to_index]
        to_start_index = to_entity["sub_token_start"]
        to_end_index = to_entity["sub_token_end"]

        if self.config.sym:
            if from_start_index > to_start_index:
                from_start_index, to_start_index = to_start_index, from_start_index
            if from_end_index > to_end_index:
                from_end_index, to_end_index = to_end_index, from_end_index
        return from_start_index, to_start_index, from_end_index, to_end_index

    def get_seperate_label_matrix(
        self, label_matrix, relations_info: List, entities_id_info_map: Dict[int, Dict]
    ):
        """separate different type of relations to different matrix(as one dimension)

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
        label_matrix = np.expand_dims(label_matrix, 0).repeat(2 * label_types, 0)

        for relation_info in relations_info:
            label_id = self.vocab.word2idx[relation_info["labels"][0]]
            (
                from_start_index,
                to_start_index,
                from_end_index,
                to_end_index,
            ) = self._get_entities_index(relation_info, entities_id_info_map)
            if from_start_index is None:
                if self.config.strict:
                    logger.warning(
                        f"Cannot get the relation of {json.dumps(relation_info)}, we will drop this instance"
                    )
                    return None
                continue
            # NOTE: label_id ==0 is unknown
            label_matrix[(label_id - 1) * 2][from_start_index][to_start_index] = 1
            label_matrix[(label_id - 1) * 2 + 1][from_end_index][to_end_index] = 1
        return label_matrix

    def get_universal_label_matrix(
        self, label_matrix, relations_info: List, entities_id_info_map: Dict[int, Dict]
    ):
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
            label_id = self.vocab.word2idx[relation_info["labels"][0]]
            (
                from_start_index,
                to_start_index,
                from_end_index,
                to_end_index,
            ) = self._get_entities_index(relation_info, entities_id_info_map)
            if from_start_index is None:
                if self.config.strict:
                    logger.warning(
                        f"Cannot get the relation of {json.dumps(relation_info)}, we will drop this instance"
                    )
                    return None
                continue
            label_matrix[0][from_start_index][to_start_index] = label_id
            label_matrix[1][from_end_index][to_end_index] = label_id
        return label_matrix
