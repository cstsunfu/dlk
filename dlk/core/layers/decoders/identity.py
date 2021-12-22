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
from . import decoder_register, decoder_config_register
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import torch


@decoder_config_register('identity')
class IdentityDecoderConfig(BaseModuleConfig):
    """Config for IdentityDecoder

    Config Example:
        >>> {
        >>>     "config": {
        >>>     },
        >>>     "_name": "identity",
        >>> }
    """
    def __init__(self, config):
        super(IdentityDecoderConfig, self).__init__(config)
        self.post_check(config['config'])


@decoder_register('identity')
class IdentityDecoder(SimpleModule):
    """Do nothing
    """

    def __init__(self, config: IdentityDecoderConfig):
        super().__init__(config)
        self.config = config

    def forward(self, inputs):
        """return inputs

        Args:
            inputs: anything

        Returns: 
            inputs 

        """
        
        return inputs
