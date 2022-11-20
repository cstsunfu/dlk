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
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import torch


@embedding_config_register('identity')
class IdentityEmbeddingConfig(BaseModuleConfig):
    default_config = {
        "_name": "identity",
        "config": {
            "output_map": {},
            "input_map": {},
            },
    }
    """Config for IdentityEmbedding

    Config Example:
        default_config
    """
    def __init__(self, config):
        super(IdentityEmbeddingConfig, self).__init__(config)
        self.post_check(config['config'])


@embedding_register('identity')
class IdentityEmbedding(SimpleModule):
    """Do nothing
    """

    def __init__(self, config: IdentityEmbeddingConfig):
        super().__init__(config)
        self.config = config

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """return inputs

        Args:
            inputs: anything

        Returns: 
            inputs 

        """
        
        return inputs
