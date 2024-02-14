# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from functools import partial
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
from tokenizers import Tokenizer

from dlk.utils.io import open
from dlk.utils.register import register

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "token_norm")
class TokenNormConfig(BaseSubProcessorConfig):
    """the token norm subprocessor"""

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

    zero_digits_replaced = BoolField(
        value=True, help="replace the digits to 0, like 1234 -> 0000"
    )
    lowercase = BoolField(value=True, help="lowercase the tokens")
    tokenizer_path = StrField(
        value="tokenizer.json",
        suggestions=["tokenizer.json"],
        help="the tokenizer of tokens",
    )

    class InputMap:
        sentence = StrField(
            value="sentence",
            suggestions=["sentence"],
            help="the input sentence",
        )

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        norm_sentence = StrField(
            value="norm_sentence",
            suggestions=["norm_sentence"],
            help="the normed sentence",
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )


@register("subprocessor", "token_norm")
class TokenNorm(BaseSubProcessor):
    """
    This part could merged to fast_tokenizer(it will save some time), but not all process need this part(except some special dataset like conll2003), and will make the fast_tokenizer be heavy.

    Token norm:
        Love -> love
        3281 -> 0000
    """

    def __init__(self, stage: str, config: TokenNormConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config

        with open(os.path.join(self.config.tokenizer_path), "r", encoding="utf-8") as f:
            tokenizer_str = json.dumps(json.load(f))

        self.tokenizer = Tokenizer.from_str(tokenizer_str)
        self.vocab = self.tokenizer.get_vocab()
        self.prefix = self.tokenizer.model.continuing_subword_prefix
        self.unk = self.tokenizer.model.unk_token

        self._zero_digits_replaced_num = 0
        self._lower_case_num = 0
        self._lower_case_zero_digits_replaced_num = 0
        logger.info(
            f"We use zero digits to replace digit token num is {self._zero_digits_replaced_num}, do lowercase token num is {self._lower_case_num}, do both num is {self._lower_case_zero_digits_replaced_num}"
        )

    def token_norm(self, token: str) -> str:
        """norm token, the result len(result) == len(token), exp.  12348->00000

        Args:
            token: origin token

        Returns:
            normed_token

        """
        if token in self.vocab:
            return token

        if self.config.zero_digits_replaced:
            norm = ""
            digit_num = 0
            for c in token:
                if c.isdigit() or c == ".":
                    norm += "0"
                    digit_num += 1
                else:
                    norm += c
            if norm in self.vocab or self.prefix + norm in self.vocab:
                self._zero_digits_replaced_num += 1
                return norm

        if self.config.lowercase:
            norm = token.lower()
            if norm in self.vocab or self.prefix + norm in self.vocab:
                self._lower_case_num += 1
                return norm

        if self.config.lowercase and self.config.zero_digits_replaced:
            norm = ""
            for c in token.lower():
                if c.isdigit() or c == ".":
                    norm += "0"
                else:
                    norm += c
            if norm in self.vocab or self.prefix + norm in self.vocab:
                self._lower_case_zero_digits_replaced_num += 1
                return norm
        return ""

    def seq_norm(self, key: str, one_item: pd.Series) -> str:
        """norm a sentence, the sentence is from one_item[key]

        Args:
            key: the name in one_item
            one_item: a pd.Series which include the key

        Returns:
            norm_sentence

        """
        seq = one_item[key]
        norm_seq = [c for c in seq]
        encode = self.tokenizer.encode(seq)
        for i, token in enumerate(encode.tokens):
            if token == self.unk:
                token_offset = encode.offsets[i]
                prenorm_token: str = seq[token_offset[0] : token_offset[1]]
                norm_token = self.token_norm(prenorm_token)
                if not norm_token:
                    continue
                assert (
                    len(norm_token) == token_offset[1] - token_offset[0]
                ), f"Prenorm '{prenorm_token}', postnorm: '{norm_token}' and {len(norm_token)}!= {token_offset[1]} - {token_offset[0]}"
                norm_seq[token_offset[0] : token_offset[1]] = norm_token
        return "".join(norm_seq)

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
        _seq_norm = partial(self.seq_norm, self.config.input_map.sentence)
        data[self.config.output_map.norm_sentence] = data.apply(_seq_norm, axis=1)
        # WARNING: if you change the apply to parallel_apply, you should change the _zero_digits_replaced_num, etc. to multiprocess safely(BTW, use parallel_apply in tokenizers==0.10.3 will make the process very slow)
        # data_set[value] = data_set.apply(_seq_norm, axis=1)

        return data
