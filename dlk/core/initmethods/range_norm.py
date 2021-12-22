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
from dlk.utils.config import BaseConfig
from . import initmethod_register, initmethod_config_register
from typing import Dict, List
import torch


@initmethod_config_register('range_norm')
class RangeNormInitConfig(BaseConfig):
    """Config for RangeNormInit

    Config Example:
        >>> {
        >>>     "_name": "range_norm",
        >>>     "config": {
        >>>         "range": 0.1,
        >>>     }
        >>> }
    """
    def __init__(self, config):
        super(RangeNormInitConfig, self).__init__(config)
        self.range = config.get("range", 0.1)
        self.post_check(config['config'], used=['range'])

@initmethod_register('range_norm')
class RangeNormInit(object):
    """default for transformers init method
    """

    def __init__(self, config: RangeNormInitConfig):
        super().__init__()
        self.range = config.range


    def __call__(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.range)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.range)
        elif isinstance(module, nn.Conv3d):
            module.weight.data.normal_(mean=0.0, std=self.range)
