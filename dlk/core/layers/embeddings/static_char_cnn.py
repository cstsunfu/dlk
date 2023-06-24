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
import pickle as pkl
from dlk.utils.io import open
from dlk.utils.logger import Logger
import torch
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules, ConfigTool


logger = Logger.get_logger()


@config_register("embedding", 'static_char_cnn')
@define
class StaticCharCNNEmbeddingConfig(BaseConfig):
    name = NameField(value="static_char_cnn", file=__file__, help="the static_char_cnn embedding module")
    @define
    class Config:
        embedding_file = StrField(value="*@*", help="the embedding file, must be saved as numpy array by pickle")
        embedding_trace = StrField(value=".", help="default the file itself, the trace of the embedding, `meta.char_embedding`,  this means the embedding = pickle.load(embedding_file)['meta']['char_embedding']")
        embedding_dim = IntField(value="*@*", checker=int_check(lower=0), help="the embedding dim")
        freeze = BoolField(value=False, help="whether to freeze the embedding")
        dropout = FloatField(value=0, checker=float_check(lower=0, upper=1), help="dropout rate")
        padding_idx = IntField(value=0, checker=int_check(lower=0), help="padding index")
        kernel_sizes = ListField(value=[3], checker=int_check(lower=0), help="the kernel sizes of the cnn")
        output_map = DictField(value={
            "char_embedding": "char_embedding",
            }, help="the output map of the char embedding module")
        input_map = DictField(value={
            "char_ids": "char_ids",
            }, help="the input map of the char embedding module")

    config = NestField(value=Config, converter=nest_converter)
    submods = SubModules(value={
        "module@cnn": "conv1d"
        }, help="the cnn module of the char embedding")
    links = DictField(value={
        "config.embedding_dim": ["module@cnn.config.in_channels", "module@cnn.config.out_channels"],
        "config.kernel_sizes": ["module@cnn.config.kernel_sizes"],
        }, help="the link of the char embedding module")


@register("embedding", 'static_char_cnn')
class StaticCharCNNEmbedding(SimpleModule):
    """ from 'char_ids' generate 'embedding'
    """
    def __init__(self, config: StaticCharCNNEmbeddingConfig):
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
        config_dict = config.to_dict()
        cnn, cnn_config = ConfigTool.get_leaf_module(register, config_register, 'module', config_dict['module@cnn'])
        self.cnn = cnn(cnn_config)
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_file, dtype=torch.float), freeze=self.config.freeze, padding_idx=0)
        assert self.embedding.weight.shape[-1] == self.config.embedding_dim

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.cnn.init_weight(method)
        logger.info(f'The static embedding is loaded the pretrained.')

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """ fit the char embedding to cnn and pool to word_embedding

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        char_ids = inputs[self.get_input_name('char_ids')]
        char_mask = (char_ids == 0).bool()
        char_embedding = self.embedding(char_ids)
        bs, seq_len, token_len, emb_dim = char_embedding.shape
        char_embedding = char_embedding.view(bs*seq_len, token_len, emb_dim)
        char_embedding = char_embedding.transpose(1, 2)
        char_embedding = self.cnn(char_embedding) # bs*seq_len, emb_dim, token_len
        word_embedding = char_embedding.masked_fill_(char_mask.view(bs*seq_len, 1, token_len), -1000).max(-1)[0].view(bs, seq_len, emb_dim).contiguous()

        inputs[self.get_output_name('char_embedding')] = self.dropout(word_embedding)
        return inputs
