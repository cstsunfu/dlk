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
from typing import Dict, List
import torch
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("initmethod", 'range_uniform')
@define
class RangeUniformInitConfig(BaseConfig):
    name = NameField(value="default", file=__file__, help="the default init method for the modules")
    @define
    class Config:
        range_from = FloatField(value=-0.1, checker=float_check(), help="the lower bound of the init value")
        range_to = FloatField(value=0.1, checker=float_check(), help="the upper bound of the init value")
    config = NestField(value=Config, converter=nest_converter)


@register("initmethod", 'range_uniform')
class RangeUniformInit(object):
    """for transformers
    """

    def __init__(self, config: RangeUniformInitConfig):
        super().__init__()
        self.config = config.config

    def __call__(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
        elif isinstance(module, nn.Conv3d):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
