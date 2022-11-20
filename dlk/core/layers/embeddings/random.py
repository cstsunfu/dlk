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

from . import embedding_register, embedding_config_register
from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import pickle as pkl
import torch.nn as nn
import torch
import numpy as np


@embedding_config_register('random')
class RandomEmbeddingConfig(BaseModuleConfig):
    """Config for RandomEmbedding

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "vocab_size": "*@*",
        >>>         "embedding_dim": "*@*",
        >>>         "dropout": 0, //dropout rate
        >>>         "padding_idx": 0, //dropout rate
        >>>         "output_map": {},
        >>>         "input_map": {},
        >>>     },
        >>>     "_name": "random",
        >>> }
    """
    def __init__(self, config: Dict):
        super(RandomEmbeddingConfig, self).__init__(config)
        config = config['config']
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.dropout = config['dropout']
        self.padding_idx = config['padding_idx']
        self.post_check(config, used=[
            "vocab_size",
            "embedding_dim",
            "padding_idx",
            "dropout",
            "return_logits",
        ])


@embedding_register('random')
class RandomEmbedding(SimpleModule):
    """ from 'input_ids' generate 'embedding'
    """

    def __init__(self, config: RandomEmbeddingConfig):
        super().__init__(config)
        self._provided_keys = set() # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = {'embedding'} # provide by this module
        self._required_keys = {'input_ids'} # required by this module
        self.config = config
        self.dropout = nn.Dropout(float(self.config.dropout))
        normal =  torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([2.0/self.config.embedding_dim]))
        self.embedding = nn.Embedding.from_pretrained(normal.sample((self.config.vocab_size, self.config.embedding_dim)), padding_idx=self.config.padding_idx)

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.embedding.apply(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """get the random embedding

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        inputs[self.get_output_name('embedding')] = self.dropout(self.embedding(inputs[self.get_input_name('input_ids')]))
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('embedding')]]))

        return inputs
