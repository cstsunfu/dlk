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
from . import embedding_register, embedding_config_register
from dlk.core.modules import module_register, module_config_register
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import pickle as pkl
from dlk.utils.io import open
from dlk.utils.logger import Logger
import torch

logger = Logger.get_logger()

@embedding_config_register('static_char_cnn')
class StaticCharCNNEmbeddingConfig(BaseModuleConfig):
    default_config = {
        "_name": "static_char_cnn",
        "config": {
            "embedding_file": "*@*", # the embedding file, must be saved as numpy array by pickle

            # if the embedding_file is a dict, you should provide the dict trace to embedding
            "embedding_trace": ".", # default the file itself is the embedding
            # embedding_trace: "char_embedding", # this means the <embedding = pickle.load(embedding_file)["char_embedding"]
            # embedding_trace: "meta.char_embedding", # this means the <embedding = pickle.load(embedding_file)['meta']["char_embedding"]
            "freeze": False, # is freeze
            "dropout": 0, # dropout rate
            "embedding_dim": 35, # dropout rate
            "kernel_sizes": [3], # dropout rate
            "padding_idx": 0,
            "output_map": {"char_embedding": "char_embedding"},
            "input_map": {"char_ids": "char_ids"},
            },
        "_link":{
            "config.embedding_dim": ["module@cnn.config.in_channels", "module@cnn.config.out_channels"],
            "config.kernel_sizes": ["module@cnn.config.kernel_sizes"],
            },
        "module@cnn": {
            "_base": "conv1d",
            "config": {
                "in_channels": -1,
                "out_channels": -1,  # will update while load embedding
                "kernel_sizes": [3],
                },
            },
    }
    """Config for StaticCharCNNEmbedding

    Config Example:
        default_config
    """
    def __init__(self, config: Dict):
        super(StaticCharCNNEmbeddingConfig, self).__init__(config)
        self.cnn_module_name: str = config['module@cnn']['_name']
        self.cnn_config = module_config_register.get(self.cnn_module_name)(config['module@cnn'])

        config = config['config']
        embedding_file = config['embedding_file']
        with open(embedding_file, 'rb') as f:
            embedding_file = pkl.load(f)
        embedding_trace = config["embedding_trace"]
        if embedding_trace != '.':
            traces = embedding_trace.split('.')
            for trace in traces:
                embedding_file = embedding_file[trace]
        self.embedding = embedding_file
        self.embedding_dim = config['embedding_dim']
        self.freeze = config['freeze']
        self.dropout = config['dropout']
        self.post_check(config, used=[
            "embedding_file",
            "embedding_trace",
            "freeze",
            "dropout",
            "embedding_dim",
            "kernel_sizes",
            "padding_idx",
            "return_logits",
        ])


@embedding_register('static_char_cnn')
class StaticCharCNNEmbedding(SimpleModule):
    """ from 'char_ids' generate 'embedding'
    """

    def __init__(self, config: StaticCharCNNEmbeddingConfig):
        super().__init__(config)
        self._provided_keys = set() # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = {'char_embedding'} # provide by this module
        self._required_keys = {'char_ids'} # required by this module
        self.config = config
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.config.embedding, dtype=torch.float), freeze=self.config.freeze, padding_idx=0)
        assert self.embedding.weight.shape[-1] == self.config.embedding_dim
        self.cnn = module_register.get(self.config.cnn_module_name)(self.config.cnn_config)

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
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('char_embedding')]]))
        return inputs
