# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Union

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


@cregister("subprocessor", "token_gather")
class TokenGatherConfig(BaseSubProcessorConfig):
    """the token gather subprocessor"""

    collect_data_set = ListField(
        value=["train"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    gather_columns = ListField(
        value=MISSING,
        suggestions=[
            ["tokens"],
            [
                "tokens",
                {"column": "entities_column", "trace": "entities_info.labels"},
            ],
        ],
        help="List of columns. If the column is a reprent as a dict,  we will trace the real elements by 'trace'.",
    )
    token_vocab = StrField(
        value="token_vocab.json",
        help="save Vocabulary object (the Vocabulary of labels) file.",
    )
    ignore = StrField(
        value="",
        help="ignore the token, the id of this token will be -100, null means nothing should be ignore",
    )
    update = StrField(
        value="",
        help="null or another exists Vocabulary object should be update",
    )
    unk = StrField(value="[UNK]", additions=[None], help="the unk token")
    pad = StrField(value="[PAD]", additions=[None], help="the pad token")
    min_freq = IntField(
        value=1,
        minimum=1,
        help="the min freq of token",
    )
    most_common = IntField(
        value=-1,
        minimum=-1,
        help="the most common token, -1 for all",
    )


@register("subprocessor", "token_gather")
class TokenGather(BaseSubProcessor):
    """Gathers tokens sequentially to build and save a global vocabulary."""

    # Declare stateful mode to ensure sequential main-process execution
    PROCESSOR_MODE = "gather"

    def __init__(self, stage: str, config: TokenGatherConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.update_path = (
            os.path.join(self.meta_dir, self.config.update)
            if self.config.update
            else None
        )
        self.vocab = None

    def _init_vocab_if_needed(self):
        """Initializes the vocabulary object on the first batch."""
        if self.vocab is None:
            if self.update_path:
                with open(self.update_path, mode="r", encoding="utf-8") as f:
                    self.vocab = Vocabulary.load(json.load(f))
            else:
                self.vocab = Vocabulary(
                    do_strip=True, unknown=self.config.unk, ignore=self.config.ignore
                )

    def get_elements_from_list_by_trace(self, data: List[Any], trace: str) -> List[Any]:
        def recursive_extract(item: Any, path_parts: List[str]) -> Any:
            if not path_parts:
                return item
            current_key = path_parts[0]
            if isinstance(item, dict):
                return recursive_extract(item.get(current_key, []), path_parts[1:])
            elif isinstance(item, (list, tuple)):
                return [recursive_extract(sub_item, path_parts) for sub_item in item]
            return item

        return [recursive_extract(item, trace.split(".")) for item in data]

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """Accumulates tokens from the current batch into the global vocabulary."""
        self._init_vocab_if_needed()

        for column in self.config.gather_columns:
            if isinstance(column, str) and column in batch:
                self.vocab.auto_update(batch[column])
            elif isinstance(column, dict):
                col_name = column["column"]
                if col_name in batch:
                    extracted = self.get_elements_from_list_by_trace(
                        batch[col_name], trace=column["trace"]
                    )
                    self.vocab.auto_update(extracted)

        return batch  # Return unmodified batch

    def save_meta(self):
        """Filters rare tokens and persists the vocabulary to disk."""
        if self.vocab is not None:
            self.vocab.filter_rare(self.config.min_freq, self.config.most_common)

            vocab_path = os.path.join(self.meta_dir, self.config.token_vocab)
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.vocab.dumps(), f)
            logger.info(
                f"Updated vocab saved to {vocab_path}. Final size: {self.vocab.word_num}"
            )
