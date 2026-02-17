# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Union

from intc import MISSING, Base, IntField, ListField, StrField, cregister

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "char_gather")
class CharGatherConfig(BaseSubProcessorConfig):
    """the char gather subprocessor"""

    collect_data_set = ListField(
        value=["train"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    gather_columns = ListField(
        value=MISSING,
        help="List of columns. Every cell must be sigle token or list of tokens or set of tokens",
    )
    char_vocab = StrField(
        value="char_vocab.json",
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
    unk = StrField(value="[UNK]", help="the unk token")
    pad = StrField(value="[PAD]", help="the pad token")
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


@register("subprocessor", "char_gather")
class CharGather(BaseSubProcessor):
    """Gathers characters sequentially to build and save a global character vocabulary."""

    PROCESSOR_MODE = "gather"

    def __init__(self, stage: str, config: CharGatherConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.update_path = (
            os.path.join(self.meta_dir, self.config.update)
            if self.config.update
            else None
        )
        self.vocab = None

    def _init_vocab_if_needed(self):
        if self.vocab is None:
            if self.update_path:
                with open(self.update_path, mode="r", encoding="utf-8") as f:
                    self.vocab = Vocabulary.load(json.load(f))
            else:
                self.vocab = Vocabulary(
                    do_strip=True, unknown=self.config.unk, ignore=self.config.ignore
                )

    def split_to_char(self, input_data: Union[str, Iterable]):
        """Recursively splits strings into distinct characters."""
        if isinstance(input_data, str):
            return list(input_data)
        elif isinstance(input_data, (list, tuple, set)):
            return [self.split_to_char(sub_input) for sub_input in input_data]
        return []

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        self._init_vocab_if_needed()

        for column in self.config.gather_columns:
            if column in batch:
                chars_batch = [self.split_to_char(item) for item in batch[column]]
                self.vocab.auto_update(chars_batch)
        return batch

    def save_meta(self):
        if self.vocab is not None:
            self.vocab.filter_rare(self.config.min_freq, self.config.most_common)

            vocab_path = os.path.join(self.meta_dir, self.config.char_vocab)
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.vocab.dumps(), f)
            logger.info(
                f"Char Vocab saved to {vocab_path}. Size: {self.vocab.word_num}"
            )
