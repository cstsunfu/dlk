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
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule
from dlk.utils.io import open
import pickle as pkl
from dlk.utils.logger import Logger
import torch
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules
from dlk.core.layers.embeddings.random import RandomEmbeddingConfig, RandomEmbedding

logger = Logger.get_logger()


@config_register("embedding", 'static')
@define
class StaticEmbeddingConfig(RandomEmbeddingConfig):
    name = NameField(value="static", file=__file__, help="the static embedding module")
    @define
    class Config(RandomEmbeddingConfig.Config):
        embedding_file = StrField(value="*@*", help="the embedding file, must be saved as numpy array by pickle")
        embedding_trace = StrField(value=".", help="default the file itself, the trace of the embedding, `meta.embedding`,  this means the embedding = pickle.load(embedding_file)['meta']['embedding']")
        freeze = BoolField(value=False, help="whether to freeze the embedding")
    config = NestField(value=Config, converter=nest_converter)

@register("embedding", 'static')
class StaticEmbedding(SimpleModule):
    """ from 'input_ids' generate static 'embedding' like glove, word2vec
    """

    def __init__(self, config: StaticEmbeddingConfig):
        super().__init__(config)
        self.config = config.config
        embedding_file = self.config.embedding_file
        with open(embedding_file, 'rb') as f:
            embedding_file = pkl.load(f)
        embedding_trace = self.config.embedding_trace
        if embedding_trace != '.':
            traces = embedding_trace.split('.')
            for trace in traces:
                embedding_file = embedding_file[trace]
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_file, dtype=torch.float), freeze=self.config.freeze, padding_idx=self.config.padding_idx)
        assert self.embedding.weight.shape[-1] == self.config.embedding_dim

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        logger.info(f'The static embedding is loaded the pretrained.')

