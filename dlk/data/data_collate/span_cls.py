# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
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
from torch.nn.utils.rnn import pad_sequence

from dlk.data.data_collate.default import DefaultCollate, DefaultCollateConfig
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary


@cregister("data_collate", "span_cls_relation")
class SpanClsRelationCollateConfig(DefaultCollateConfig):
    """
    Config for collating span classification and relation extraction tasks.
    Handles sparse labels and converts them to dense multi-label tensors.
    """

    label_ids = StrField(
        value="label_ids",
        help="Key for the sparse entity labels, e.g., 'label_ids'",
    )
    num_entity_classes = IntField(value=MISSING, help="Total number of entity classes.")

    head_label_ids = StrField(
        value="head_label_ids",
        help="Key for the sparse head-to-head relation labels, e.g., 'head_label_ids'",
    )
    tail_label_ids = StrField(
        value="tail_label_ids",
        help="Key for the sparse tail-to-tail relation labels, e.g., 'tail_label_ids'",
    )
    num_relation_classes = IntField(value=0, help="Total number of relation classes.")
    predict = BoolField(
        value=False, help="Whether this collate is used for prediction."
    )


@register("data_collate", "span_cls_relation")
class SpanClsRelationCollate(DefaultCollate):
    """
    A specialized collate function for span classification and relation tasks.
    It inherits from DefaultCollate to handle standard padding (input_ids, etc.)
    and adds functionality to convert sparse span/relation labels into dense tensors
    suitable for multi-label classification.
    """

    def __init__(self, config: SpanClsRelationCollateConfig):
        super().__init__(config)
        self.config = config  # self.config is now of type SpanClsRelationCollateConfig
        self.sparse_keys = {
            "entity": self.config.label_ids,
            "head": self.config.head_label_ids,
            "tail": self.config.tail_label_ids,
        }

    def _sparse_to_dense(
        self,
        sparse_labels: List[List[List[int]]],
        batch_size: int,
        seq_len: int,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Converts a batch of sparse labels into a dense tensor.

        Args:
            sparse_labels: A list of lists, where each inner list contains [from_idx, to_idx, label_id].
            batch_size: The number of items in the batch.
            seq_len: The maximum sequence length in the batch.
            num_classes: The total number of classes for this label type.

        Returns:
            A dense tensor of shape (batch_size, num_classes, seq_len, seq_len).
        """
        target = torch.zeros(
            (batch_size, num_classes, seq_len, seq_len), dtype=torch.float
        )

        for i, sample_labels in enumerate(sparse_labels):
            for from_idx, to_idx, label_id in sample_labels:
                target[i, label_id, from_idx, to_idx] = 1.0
        return target

    def __call__(
        self, batch: List[Dict], stage: str = "predict"
    ) -> Dict[str, torch.Tensor]:
        # Step 1: Separate the sparse labels from the rest of the data.
        sparse_data = {k: [] for k, v in self.sparse_keys.items() if v}
        standard_batch = []

        for sample in batch:
            standard_sample = sample.copy()
            for k, key_name in self.sparse_keys.items():
                sparse_name = "sparse_" + key_name
                if sparse_name in standard_sample:
                    sparse_data[k].append(standard_sample.pop(sparse_name))
            standard_batch.append(standard_sample)

        # Step 2: Use the parent's __call__ to handle all standard padding.
        collated_data = super().__call__(standard_batch)

        if stage in {"online", "predict", "serve"}:
            return collated_data

        # Step 3: Process the separated sparse labels into dense tensors.
        batch_size, seq_len = collated_data["input_ids"].shape

        dense_entity_labels = self._sparse_to_dense(
            sparse_labels=sparse_data["entity"],
            batch_size=batch_size,
            seq_len=seq_len,
            num_classes=self.config.num_entity_classes,
        )
        collated_data[self.config.label_ids] = dense_entity_labels

        if self.config.num_relation_classes:
            dense_head_labels = self._sparse_to_dense(
                sparse_labels=sparse_data["head"],
                batch_size=batch_size,
                seq_len=seq_len,
                num_classes=self.config.num_relation_classes,
            )
            collated_data[self.config.head_label_ids] = dense_head_labels

            dense_tail_labels = self._sparse_to_dense(
                sparse_labels=sparse_data["tail"],
                batch_size=batch_size,
                seq_len=seq_len,
                num_classes=self.config.num_relation_classes,
            )
            collated_data[self.config.tail_label_ids] = dense_tail_labels

        return collated_data
