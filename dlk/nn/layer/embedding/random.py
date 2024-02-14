# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl
from typing import Callable, Dict, List, Set

import numpy as np
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
from dlk.utils.register import register


@cregister("embedding", "random")
class RandomEmbeddingConfig(Base):
    """
    the random embedding module config
    """

    vocab_size = IntField(
        value=0,
        minimum=0,
        help="the vocab size, when the vocab size is 0, it means this class is overload by other Embedding(like Static)",
    )
    embedding_dim = IntField(value=MISSING, minimum=1, help="the embedding dim")
    dropout = FloatField(value=0, minimum=0.0, maximum=1.0, help="dropout rate")
    padding_idx = IntField(value=0, minimum=0, help="padding index")

    class OutputMap:
        embedding = StrField(value="embedding", help="the output of embedding name")

    output_map = NestField(
        value=OutputMap, help="the output map of the random embedding module"
    )

    class InputMap:
        input_ids = StrField(value="input_ids", help="the input of input_ids name")

    input_map = NestField(
        value=InputMap, help="the input map of the random embedding module"
    )


@register("embedding", "random")
class RandomEmbedding(SimpleModule):
    """from 'input_ids' generate 'embedding'"""

    def __init__(self, config: RandomEmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.dropout = nn.Dropout(float(self.config.dropout))
        normal = torch.distributions.Normal(
            torch.tensor([0.0]), torch.tensor([2.0 / self.config.embedding_dim])
        )
        if self.config.vocab_size:
            self.embedding = nn.Embedding.from_pretrained(
                normal.sample(
                    (self.config.vocab_size, self.config.embedding_dim)
                ).squeeze_(-1),
                padding_idx=self.config.padding_idx,
            )

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.embedding.apply(method)

    def share_embedding(self, embedding):
        """link the embedding.embedding to self.embedding

        Args:
            embedding: source embedding

        Returns:
            None

        """
        self.embedding = embedding.embedding

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """get the random embedding

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        inputs[self.config.output_map.embedding] = self.dropout(
            self.embedding(inputs[self.config.input_map.input_ids])
        )
        return inputs
