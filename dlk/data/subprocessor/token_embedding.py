# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
from intc import (
    MISSING,
    Base,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    cregister,
)

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.tokenizer_util import load_fast_tokenizer
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "token_embedding")
class TokenEmbeddingConfig(BaseSubProcessorConfig):
    """the token embedding subprocessor"""

    collect_data_set = ListField(
        value=["train"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    embedding_file = StrField(
        value=MISSING,
        suggestions=["embedding_file"],
        help="the embedding file path",
    )
    tokenizer_path = StrField(
        value=None,
        suggestions=["tokenizer"],
        additions=[None],
        help="the tokenizer path",
    )
    vocab = StrField(
        value=None,
        suggestions=["vocab"],
        additions=[None],
        help="the vocab path",
    )
    token_embedding = StrField(
        value="token_embedding",
        suggestions=["token_embedding"],
        help="the embedding saved name",
    )
    embedding_size = IntField(value=MISSING, minimum=1, help="the embedding size")

    class BiasClipRange:
        lower = FloatField(value=-10e9, help="lower bound")
        upper = FloatField(value=10e9, help="upper bound")

    bias_clip_range = NestField(value=BiasClipRange, help="bias clip range")


@register("subprocessor", "token_embedding")
class TokenEmbedding(BaseSubProcessor):
    """
    Gather tokens embedding from pretrained 'embedding_file' or init embedding.
    """

    def __init__(self, stage: str, config: TokenEmbeddingConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.tokenizer = None
        self.vocab = None
        if self.config.embedding_file:
            self.origin_embedding = self.get_embedding(
                self.config.embedding_file, self.config.embedding_size
            )
        else:
            self.origin_embedding = {}

    def load_meta(self):
        self.loaded_meta = True
        if self.config.vocab:
            self.vocab = Vocabulary.load_from_file(
                os.path.join(self.meta_dir, self.config.vocab)
            )
        if self.config.tokenizer_path:
            self.tokenizer = load_fast_tokenizer(self.config.tokenizer_path)

    def get_embedding(self, file_path, embedding_size) -> Dict[str, List[float]]:
        embedding_dict = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.rstrip().split(" ")
                if len(parts) != embedding_size + 1:
                    continue
                word = parts[0]
                vector = list(map(float, parts[1:]))
                embedding_dict[word] = vector
        return embedding_dict

    def update_embedding(self, embedding_dict: Dict[str, List[float]], vocab: Dict):
        low = self.config.bias_clip_range.lower
        up = self.config.bias_clip_range.upper
        bias = np.sqrt(6.0 / (len(vocab) + self.config.embedding_size))
        bias = max(low, min(bias, up))

        for token in vocab:
            if token not in embedding_dict:
                embedding_dict[token] = list(
                    np.random.uniform(-bias, bias, self.config.embedding_size)
                )
        return embedding_dict

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        if not deliver_meta:
            return batch

        if not self.loaded_meta:
            self.load_meta()

        if self.tokenizer:
            token2id = self.tokenizer.get_vocab()
            id2token = {v: k for k, v in token2id.items()}
        elif self.vocab:
            token2id = self.vocab.word2idx
            id2token = self.vocab.idx2word
        else:
            raise PermissionError("Tokenizer or Vocab must be provided.")

        embedding_dict = self.update_embedding(self.origin_embedding, token2id)

        embedding_mat = np.zeros((len(id2token), self.config.embedding_size))
        for idx in range(len(id2token)):
            token = id2token.get(idx)
            if token in embedding_dict:
                embedding_mat[idx] = embedding_dict[token]

        save_path = os.path.join(self.meta_dir, self.config.token_embedding)
        with open(save_path, "wb") as f:
            embedding_mat.dump(f)

        logger.info(f"Saved embedding matrix to {save_path}")

        return batch
