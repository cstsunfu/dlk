# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule
import pickle as pkl
import torch.nn as nn
import torch
import numpy as np
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("embedding", 'random')
@define
class RandomEmbeddingConfig(BaseConfig):
    name = NameField(value="random", file=__file__, help="the random embedding module")
    @define
    class Config:
        vocab_size = IntField(value="*@*", checker=int_check(lower=0), help="the vocab size")
        embedding_dim = IntField(value="*@*", checker=int_check(lower=0), help="the embedding dim")
        dropout = FloatField(value=0, checker=float_check(lower=0, upper=1), help="dropout rate")
        padding_idx = IntField(value=0, checker=int_check(lower=0), help="padding index")
        output_map = DictField(value={
            "embedding": "embedding",
            }, help="the output map of the random embedding module")
        input_map = DictField(value={
            "input_ids": "input_ids",
            }, help="the input map of the random embedding module")

    config = NestField(value=Config, converter=nest_converter)


@register("embedding", 'random')
class RandomEmbedding(SimpleModule):
    """ from 'input_ids' generate 'embedding'
    """

    def __init__(self, config: RandomEmbeddingConfig):
        super().__init__(config)
        self.config = config.config
        self.dropout = nn.Dropout(float(self.config.dropout))
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([2.0/self.config.embedding_dim]))
        self.embedding = nn.Embedding.from_pretrained(normal.sample((self.config.vocab_size, self.config.embedding_dim)).squeeze_(-1), padding_idx=self.config.padding_idx)

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

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """get the random embedding

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        inputs[self.get_output_name('embedding')] = self.dropout(self.embedding(inputs[self.get_input_name('input_ids')]))
        return inputs
