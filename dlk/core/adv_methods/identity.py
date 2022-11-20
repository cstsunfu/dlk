# Copyright cstsunfu. All rights reserved.
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
import torch
from . import adv_method_register, adv_method_config_register, AdvMethod
from typing import Dict, List
from dlk.utils.logger import Logger

logger = Logger.get_logger()

@adv_method_config_register('identity')
class IdentityAdvMethodConfig(object):
    default_config = {
        "_name": "identity",
        "config": {
        }
    }
    """Config for IdentityAdvMethod

    Config Example:
        default_config
    """
    def __init__(self, config: Dict):
        super(IdentityAdvMethodConfig, self).__init__()
        config = config['config']

@adv_method_register('identity')
class IdentityAdvMethod(AdvMethod):
    """Save identity decided by config
    """

    def __init__(self, model: nn.Module, config: IdentityAdvMethodConfig):
        super().__init__(model, config)
        pass


    def training_step(self, imodel, batch: Dict[str, torch.Tensor], batch_idx: int):
        """do training_step on a mini batch

        Args:
            imodel: imodel instance
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns: 
            the outputs

        """
        raise NotImplementedError("You should not call the identity adv method")
