# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from functools import partial
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


@cregister("subprocessor", "token2charid")
class Token2CharIDConfig(BaseSubProcessorConfig):
    """the token 2 character id subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict = ListField(
        value=["predict"],
        suggestions=[["predict"]],
        help="the data set should be processed for predict stage",
    )
    online = ListField(
        value=["online"],
        suggestions=[["online"]],
        help="the data set should be processed for online stage",
    )

    class InputMap:
        sentence = StrField(value="sentence", help="the sentence")
        offsets = StrField(value="offsets", help="the offsets")

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        char_ids = StrField(value="char_ids", help="the char ids")

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )
    vocab = StrField(value="char_vocab.json", help="the vocab file for the character")
    max_token_len = IntField(
        value=20,
        minimum=1,
        help="the max length of token, then the output will be max_token_len x token_num (put max_token_len in previor is for padding on token_num)",
    )


@register("subprocessor", "token2charid")
class Token2CharID(BaseSubProcessor):
    """Use 'Vocabulary' map the character from tokens to id"""

    def __init__(self, stage: str, config: Token2CharIDConfig, meta_dir: str):
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
        """firstpiece relabel the data
        one_token like 'apple' will generate [1, 2, 2, 3] if max_token_len==4 and the vocab.word2idx = {'a': 1, "p": 2, "l": 3}

        Args:
            data: will processed data

            deliver_meta:
                ignore
        Returns:
            relabeld data
        """
        if not self.loaded_meta:
            self.load_meta()

        def get_index_wrap(sentence_name, offset_name, x):
            """wrap the vocab.get_index"""
            sentence = list(x[sentence_name])
            offsets = x[offset_name]
            char_ids = []
            for offset in offsets:
                token = sentence[offset[0] : offset[1]][: self.config.max_token_len]
                token = token + [self.vocab.pad] * (
                    self.config.max_token_len - len(token)
                )
                char_ids.append([self.vocab.get_index(c) for c in token])
            return char_ids

        get_index = partial(
            get_index_wrap,
            self.config.input_map.sentence,
            self.config.input_map.offsets,
        )
        data[self.config.output_map.char_ids] = data.apply(get_index, axis=1)
        return data
