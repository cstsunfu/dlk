# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Callable, Dict, Iterable, List, Set, Union

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


@cregister("subprocessor", "char_gather")
class CharGatherConfig(BaseSubProcessorConfig):
    """the char gather subprocessor"""

    train_data_set = ListField(
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
        help="null or another exists Vocabulary object should be update, if the update is not null, the unk and pad ignore will be ignored",
    )
    unk = StrField(value="[UNK]", help="the unk token")
    pad = StrField(value="[PAD]", help="the pad token")
    min_freq = IntField(
        value=1,
        minimum=1,
        help="the min freq of token, you can only change one of the value of min_freq and most_common",
    )
    most_common = IntField(
        value=-1,
        minimum=-1,
        help="the most common token, -1 for all, you can only change one of the value of min_freq and most_common.",
    )


@register("subprocessor", "char_gather")
class CharGather(BaseSubProcessor):
    """gather all character from the 'gather_columns' and deliver a vocab named 'char_vocab'"""

    def __init__(self, stage: str, config: CharGatherConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.update = (
            None
            if not self.config.update
            else os.path.join(self.meta_dir, self.config.update)
        )

    def split_to_char(self, input: Union[str, Iterable]):
        """the char is from token or sentence, so we need split them to List[char]

        Args:
            input: auto detach the type of input and split it to char

        Returns:
            the same shape of the input but the str is split to List[char]

        """
        if isinstance(input, str):
            return [c for c in input]
        else:
            return [self.split_to_char(sub_input) for sub_input in input]

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """Character gather entry

        Args:
            data:
            >>> |sentence |label|
            >>> |---------|-----|
            >>> |sent_a...|la   |
            >>> |sent_b...|lb   |

            deliver_meta:
                if there are some meta info need to deliver to next processor, and deliver_meta is True, save the meta info to datadir
        Returns:
            processed data

        """
        if not deliver_meta:
            return data
        if self.update:
            with open(self.update, mode="r", encoding="utf-8") as f:
                self.vocab = Vocabulary.load(json.load(f))
        else:
            self.vocab = Vocabulary(
                do_strip=True, unknown=self.config.unk, ignore=self.config.ignore
            )
        for column in self.config.gather_columns:
            column: str
            self.vocab.auto_update(self.split_to_char(data[column]))
        self.vocab.filter_rare(self.config.min_freq, self.config.most_common)
        logger.info(f"The Char Vocab Num is {self.vocab.word_num}")
        with open(
            os.path.join(self.meta_dir, self.config.char_vocab), "w", encoding="utf-8"
        ) as f:
            json.dump(self.vocab.dumps(), f)
        return data
