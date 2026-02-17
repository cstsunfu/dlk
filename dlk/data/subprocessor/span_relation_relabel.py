# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from intc import Base, BoolField, ListField, NestField, StrField, cregister

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

    input_map = NestField(value=InputMap, help="input map")

    class OutputMap:
        sparse_head_label_ids = StrField(
            value="sparse_head_label_ids",
            help="sparse head relations",
        )
        sparse_tail_label_ids = StrField(
            value="sparse_tail_label_ids",
            help="sparse tail relations",
        )

    output_map = NestField(value=OutputMap, help="output map")
    vocab = StrField(value="label_vocab#relation.json", help="relation vocab")
    sym = BoolField(value=True, help="symmetric relation")
    strict = BoolField(value=True, help="strict mode")


@register("subprocessor", "span_relation_relabel")
class SpanRelationRelabel(BaseSubProcessor):
    """Relabel relations for TPLinker-like sparse output."""

    def __init__(self, stage: str, config: SpanRelationRelabelConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.vocab: Vocabulary = None

    def load_meta(self):
        self.loaded_meta = True
        self.vocab = Vocabulary.load_from_file(
            os.path.join(self.meta_dir, self.config.vocab)
        )

    def _relabel_one(self, relations_info, processed_entities_info):
        entities_map = {e["entity_id"]: e for e in processed_entities_info}
        head_labels = []
        tail_labels = []

        for relation in relations_info:
            fid, tid = relation["from"], relation["to"]
            if fid not in entities_map or tid not in entities_map:
                if self.config.strict:
                    return None, None
                continue

            f_ent, t_ent = entities_map[fid], entities_map[tid]
            fs, fe = f_ent["sub_token_start"], f_ent["sub_token_end"]
            ts, te = t_ent["sub_token_start"], t_ent["sub_token_end"]

            if self.config.sym and fs > ts:
                fs, ts = ts, fs
                fe, te = te, fe

            # Using vocab for relation label ID
            lid = self.vocab.get_index(relation["labels"][0])
            head_labels.append([fs, ts, lid])
            tail_labels.append([fe, te, lid])

        return head_labels, tail_labels

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        if not self.loaded_meta:
            self.load_meta()

        rel_col = self.config.input_map.relations_info
        ent_col = self.config.input_map.processed_entities_info

        res_head = []
        res_tail = []
        valid_indices = []

        for i, (rels, ents) in enumerate(zip(batch[rel_col], batch[ent_col])):
            h, t = self._relabel_one(rels, ents)
            if h is None and self.config.strict:
                continue
            res_head.append(h)
            res_tail.append(t)
            valid_indices.append(i)

        if len(valid_indices) < len(batch[rel_col]):
            new_batch = {}
            for k, v in batch.items():
                new_batch[k] = [v[i] for i in valid_indices]
            batch = new_batch

        batch[self.config.output_map.sparse_head_label_ids] = res_head
        batch[self.config.output_map.sparse_tail_label_ids] = res_tail

        return batch
