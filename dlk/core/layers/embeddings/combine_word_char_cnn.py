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

import torch.nn as nn
from typing import Callable, Dict, List, Set
from dlk.core.base_module import SimpleModule
import pickle as pkl
import torch
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules, ConfigTool


@config_register("embedding", 'combine_word_char_cnn')
@define
class CombineWordCharCNNEmbeddingConfig(BaseConfig):
    name = NameField(value="combine_word_char_cnn", file=__file__, help="the combine_word_char_cnn embedding module")
    @define
    class Config:
        embedding_dim = IntField(value="*@*", checker=int_check(lower=0), help="the embedding dim")
        dropout = FloatField(value=0, checker=float_check(lower=0, upper=1), help="dropout rate")
        output_map = DictField(value={
            "embedding": "embedding",
            }, help="the output map of the char embedding module")
        input_map = DictField(value={
            "char_ids": "char_ids",
            "input_ids": "input_ids",
            }, help="the input map of the combine embedding module")

    config = NestField(value=Config, converter=nest_converter)
    submods = SubModules(value={
        "embedding@char": {
            "base": "static_char_cnn",
            "config": {
                "output_map": {"char_embedding": "char_embedding"},
                }
                           },
        "embedding@word": {
            "base": "static",
            "config": {
                "output_map": {"embedding": "word_embedding"},
                }
            },
        }, help="the char and word embedding module")
    links = DictField(value={
        "embedding@char.config.embedding_dim,embedding@word.config.embedding_dim @@lambda x, y: x+y": ["config.embedding_dim"],
        "config.dropout": ["embedding@char.config.dropout", "embedding@word.config.dropout"]
        }, help="the links between config and submodules")


@register("embedding", 'combine_word_char_cnn')
class CombineWordCharCNNEmbedding(SimpleModule):
    """ from 'input_ids' and 'char_ids' generate 'embedding'
    """
    def __init__(self, config: CombineWordCharCNNEmbeddingConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(float(config.config.dropout))
        config_dict = config.to_dict()
        self.word_embedding = ConfigTool.get_leaf_module(register, config_register, "embedding", config_dict['embedding@word'], init=True)
        self.char_embedding = ConfigTool.get_leaf_module(register, config_register, "embedding", config_dict['embedding@char'], init=True)

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.word_embedding.init_weight(method)
        self.char_embedding.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """get the combine char and word embedding

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        inputs = self.word_embedding(inputs)
        inputs = self.char_embedding(inputs)

        combine_embedding = torch.cat([inputs["char_embedding"], inputs["word_embedding"]], dim=-1)
        inputs[self.get_output_name('embedding')] = self.dropout(combine_embedding)
        return inputs
