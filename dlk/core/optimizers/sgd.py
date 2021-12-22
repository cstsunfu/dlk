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


@optimizer_config_register("sgd")
class SGDOptimizerConfig(BaseConfig):
    """Config for SGDOptimizer

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "lr": 1e-3,
        >>>         "momentum": 0.9,
        >>>         "dampening": 0,
        >>>         "weight_decay": 0,
        >>>         "nesterov":false,
        >>>         "optimizer_special_groups": {
        >>>             // "order": ['decoder', 'bias'], // the group order, if the para is in decoder & is in bias, set to decoder. The order name is set to the group name
        >>>             // "bias": {
        >>>             //     "config": {
        >>>             //         "weight_decay": 0
        >>>             //     },
        >>>             //     "pattern": ["bias",  "LayerNorm.bias", "LayerNorm.weight"]
        >>>             // },
        >>>             // "decoder": {
        >>>             //     "config": {
        >>>             //         "lr": 1e-3
        >>>             //     },
        >>>             //     "pattern": ["decoder"]
        >>>             // },
        >>>         }
        >>>         "name": "default" // default group name
        >>>     },
        >>>     "_name": "sgd",
        >>> }
    """
    def __init__(self, config: Dict):
        super(SGDOptimizerConfig, self).__init__(config)
        self.config = config['config']
        self.post_check(self.config, used=[
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "optimizer_special_groups",
        ])


@optimizer_register("sgd")
class SGDOptimizer(BaseOptimizer):
    """wrap for optim.SGD"""
    def __init__(self, model: nn.Module, config: SGDOptimizerConfig):
        super(SGDOptimizer, self).__init__()
        self.config = config.config
        self.model = model
        self.optimizer = optim.SGD

    def get_optimizer(self)->optim.SGD:
        """return the initialized SGD optimizer

        Returns: 
            SGD Optimizer

        """
        return self.init_optimizer(optim.SGD, self.model, self.config)
