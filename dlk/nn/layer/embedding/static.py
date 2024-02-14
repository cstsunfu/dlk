# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle as pkl
from typing import Dict, List, Set

import torch
import torch.nn as nn
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

from dlk.nn.base_module import SimpleModule
from dlk.nn.layer.embedding.random import RandomEmbedding, RandomEmbeddingConfig
from dlk.utils.io import open
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("embedding", "static")
class StaticEmbeddingConfig(RandomEmbeddingConfig):
    """
    the static embedding module config
    """

    embedding_file = StrField(
        value=MISSING,
        help="the embedding file, must be saved as numpy array by pickle",
    )
    freeze = BoolField(value=False, help="whether to freeze the embedding")


@register("embedding", "static")
class StaticEmbedding(RandomEmbedding):
    """from 'input_ids' generate static 'embedding' like glove, word2vec"""

    def __init__(self, config: StaticEmbeddingConfig):
        super().__init__(config)
        self.config = config
        embedding_file = self.config.embedding_file
        with open(embedding_file, "rb") as f:
            embedding_file = pkl.load(f)
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_file, dtype=torch.float),
            freeze=self.config.freeze,
            padding_idx=self.config.padding_idx,
        )
        assert (
            self.embedding.weight.shape[-1] == self.config.embedding_dim
        ), f"{self.embedding.weight.shape[-1]} != {self.config.embedding_dim}"

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        logger.info(f"The static embedding is loaded the pretrained embedding.")
