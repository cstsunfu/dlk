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
from . import BaseOptimizer, BaseOptimizerConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("optimizer", 'adamw')
@define
class AdamWOptimizerConfig(BaseOptimizerConfig):
    name = NameField(value="adamw", file=__file__, help="the adamw optimizer")
    @define
    class Config(BaseOptimizerConfig.Config):
        lr = FloatField(value=1e-3, checker=float_check(lower=0), help="the learning rate of optimizer")
        betas = ListField(value=[0.9, 0.999], checker=suggestions(), help="the betas of adamw")
        eps = FloatField(value=1e-8, checker=float_check(lower=0), help="the epsilon of optimizer")
        weight_decay = FloatField(value=1e-2, checker=float_check(lower=0), help="the weight decay of the optimizer")
    config = NestField(value=Config, converter=nest_converter)


@register("optimizer", "adamw")
class AdamWOptimizer(BaseOptimizer):
    """Wrap for optim.AdamW
    """
    def __init__(self, model: nn.Module, config: AdamWOptimizerConfig):
        super(AdamWOptimizer, self).__init__(model, config, optim.AdamW)
