# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, List

from intc import MISSING, Base, BoolField, ListField, NestField, StrField, cregister

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.tokenizer_util import load_fast_tokenizer

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
        help="the input map of the processor",
    )

    class OutputMap:
        norm_sentence = StrField(
            value="norm_sentence",
            suggestions=["norm_sentence"],
            help="the normed sentence",
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor",
    )


@register("subprocessor", "token_norm")
class TokenNorm(BaseSubProcessor):
    """Normalize tokens in a sentence (digits -> 0, lowercase) while preserving alignment."""

    def __init__(self, stage: str, config: TokenNormConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config

        self.tokenizer = load_fast_tokenizer(self.config.tokenizer_path)
        self.vocab = self.tokenizer.get_vocab()

        self.prefix = None
        if hasattr(self.tokenizer, "backend_tokenizer") and hasattr(
            self.tokenizer.backend_tokenizer.model, "continuing_subword_prefix"
        ):
            self.prefix = getattr(
                self.tokenizer.backend_tokenizer.model,
                "continuing_subword_prefix",
                None,
            )

        self.unk = self.tokenizer.unk_token

    def token_norm(self, token: str) -> str:
        """Normalize a single token string."""
        if token in self.vocab:
            return token

        norm = token
        if self.config.lowercase:
            norm = norm.lower()

        if self.config.zero_digits_replaced:
            # Replace digits with 0
            norm = "".join(["0" if c.isdigit() or c == "." else c for c in norm])

        # Check validity in vocab
        if norm in self.vocab or (self.prefix and self.prefix + norm in self.vocab):
            return norm

        return ""

    def seq_norm(self, seq: str) -> str:
        """Normalize a full sentence string based on tokenization."""
        norm_seq = list(seq)
        encodings = self.tokenizer(
            seq, return_offsets_mapping=True, add_special_tokens=False
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encodings["input_ids"])
        offsets = encodings["offset_mapping"]

        for i, token in enumerate(tokens):
            if token == self.unk:
                start, end = offsets[i]
                if start == end:
                    continue

                prenorm_token = seq[start:end]
                normed_token = self.token_norm(prenorm_token)

                if normed_token:
                    # Only replace if length matches to preserve char-level alignment
                    if len(normed_token) == (end - start):
                        norm_seq[start:end] = list(normed_token)

        return "".join(norm_seq)

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """Normalize sentences in the batch."""
        input_col = self.config.input_map.sentence
        output_col = self.config.output_map.norm_sentence

        if input_col in batch:
            batch[output_col] = [self.seq_norm(sent) for sent in batch[input_col]]

        return batch
