# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Callable, Dict, List, Set, Tuple

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
        relations_info = StrField(value="relations_info", help="the relations info")
        processed_entities_info = StrField(
            value="processed_entities_info",
            help="the processed entities info from span_cls_relabel",
        )

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        sparse_head_label_ids = StrField(
            value="sparse_head_label_ids",
            help="the sparse head-to-head relation labels in format: [[from_idx, to_idx, label_id], ...]",
        )
        sparse_tail_label_ids = StrField(
            value="sparse_tail_label_ids",
            help="the sparse tail-to-tail relation labels in format: [[from_idx, to_idx, label_id], ...]",
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )
    vocab = StrField(
        value="label_vocab#relation.json", help="the vocab for the relation label"
    )

    sym = BoolField(
        value=True,
        help="whether the from entity and end entity can swap in relations (ensures from_idx <= to_idx)",
    )
    strict = BoolField(
        value=True, help="if strict == True, will drop the invalid sample"
    )


@register("subprocessor", "span_relation_relabel")
class SpanRelationRelabel(BaseSubProcessor):
    """
    Relabel relations to two separate sparse lists for head-to-head and tail-to-tail relations,
    based on TPLinker logic.
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

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        if not self.loaded_meta:
            self.load_meta()

        data[
            [
                self.config.output_map.sparse_head_label_ids,
                self.config.output_map.sparse_tail_label_ids,
            ]
        ] = data.apply(self.relabel, axis=1, result_type="expand")

        if self.config.strict:
            # Drop rows where either of the new columns is None
            data.dropna(
                subset=[
                    self.config.output_map.sparse_head_label_ids,
                    self.config.output_map.sparse_tail_label_ids,
                ],
                inplace=True,
            )
            data.reset_index(inplace=True, drop=True)

        return data

    def relabel(self, one_ins: pd.Series) -> Tuple[List, List]:
        """
        Creates two sparse lists of labels for head-to-head and tail-to-tail relations.

        Args:
            one_ins: A pandas series containing 'relations_info' and 'processed_entities_info'.

        Returns:
            A tuple containing (head_relation_labels, tail_relation_labels).
            Returns (None, None) on failure if strict mode is on.
        """
        relations_info = one_ins[self.config.input_map.relations_info]
        processed_entities_info = one_ins[self.config.input_map.processed_entities_info]

        entities_id_info_map = {
            entity_info["entity_id"]: entity_info
            for entity_info in processed_entities_info
        }

        return self._create_sparse_relation_labels(relations_info, entities_id_info_map)

    def _get_entities_index(
        self, relation_info, entities_id_info_map: Dict
    ) -> Tuple[int, int, int, int]:
        """Gets the start/end token indices for the 'from' and 'to' entities in a relation."""
        from_entity_id = relation_info["from"]
        to_entity_id = relation_info["to"]

        if (
            from_entity_id not in entities_id_info_map
            or to_entity_id not in entities_id_info_map
        ):
            return None, None, None, None

        from_entity = entities_id_info_map[from_entity_id]
        from_start_index = from_entity["sub_token_start"]
        from_end_index = from_entity["sub_token_end"]
        to_entity = entities_id_info_map[to_entity_id]
        to_start_index = to_entity["sub_token_start"]
        to_end_index = to_entity["sub_token_end"]

        # If symmetric, enforce that the first index is always smaller to create a canonical representation
        if self.config.sym and from_start_index > to_start_index:
            from_start_index, to_start_index = to_start_index, from_start_index
            from_end_index, to_end_index = to_end_index, from_end_index

        return from_start_index, to_start_index, from_end_index, to_end_index

    def _create_sparse_relation_labels(
        self, relations_info: List, entities_id_info_map: Dict
    ) -> Tuple[List, List]:
        """
        Generates sparse lists for head-to-head and tail-to-tail relations.

        Args:
            relations_info: The list of relation dictionaries.
            entities_id_info_map: A map from entity_id to entity information.

        Returns:
            A tuple: (head_labels, tail_labels).
            `head_labels`: A list of [from_start_idx, to_start_idx, relation_id]
            `tail_labels`: A list of [from_end_idx, to_end_idx, relation_id]
            Returns (None, None) if a critical error occurs and strict mode is on.
        """
        head_labels = []
        tail_labels = []

        for relation_info in relations_info:
            label_id = self.vocab.word2idx.get(relation_info["labels"][0])
            (
                from_start_index,
                to_start_index,
                from_end_index,
                to_end_index,
            ) = self._get_entities_index(relation_info, entities_id_info_map)

            if from_start_index is None:
                if self.config.strict:
                    logger.warning(
                        f"Cannot find entity for relation {json.dumps(relation_info)}. Dropping instance."
                    )
                    return None, None  # Signal to drop the row
                continue

            head_labels.append([from_start_index, to_start_index, label_id])
            tail_labels.append([from_end_index, to_end_index, label_id])

        return head_labels, tail_labels
