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

from typing import Dict
import torch.nn as nn
import torch.optim as optim
from dlk.utils.config import BaseConfig
from . import optimizer_register, optimizer_config_register, BaseOptimizer


@optimizer_config_register("adamw")
class AdamWOptimizerConfig(BaseConfig):
    """Config for AdamWOptimizer

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "lr": 5e-5,
        >>>         "betas": [0.9, 0.999],
        >>>         "eps": 1e-6,
        >>>         "weight_decay": 1e-2,
        >>>         "optimizer_special_groups": {
        >>>             "order": ['decoder', 'bias'], // the group order, if the para is in decoder & is in bias, set to decoder. The order name is set to the group name
        >>>             "bias": {
        >>>                 "config": {
        >>>                     "weight_decay": 0
        >>>                 },
        >>>                 "pattern": ["bias",  "LayerNorm.bias", "LayerNorm.weight"]
        >>>             },
        >>>             "decoder": {
        >>>                 "config": {
        >>>                     "lr": 1e-3
        >>>                 },
        >>>                 "pattern": ["decoder"]
        >>>             },
        >>>         }
        >>>         "name": "default" // default group name
        >>>     },
        >>>     "_name": "adamw",
        >>> }
    """
    def __init__(self, config: Dict):
        super(AdamWOptimizerConfig, self).__init__(config)
        self.config = config['config']
        self.post_check(self.config, used=[
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "optimizer_special_groups",
            "name",
        ])


@optimizer_register("adamw")
class AdamWOptimizer(BaseOptimizer):
    """Wrap for optim.AdamW
    """
    def __init__(self, model: nn.Module, config: AdamWOptimizerConfig):
        super(AdamWOptimizer, self).__init__()
        self.config = config.config
        self.model = model
        self.optimizer = optim.AdamW

    def get_optimizer(self)->optim.AdamW:
        """return the initialized AdamW optimizer

        Returns: 
            AdamW Optimizer

        """
        return self.init_optimizer(optim.AdamW, self.model, self.config)
