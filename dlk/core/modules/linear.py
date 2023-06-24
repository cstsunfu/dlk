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
import torch
from typing import Dict, List
from . import Module
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

@config_register("module", 'linear')
@define
class LinearConfig(BaseConfig):
    name = NameField(value="linear", file=__file__, help="the linear config")
    @define
    class Config:
        input_size = IntField(value="*@*", checker=int_check(lower=0), help="the input size")
        output_size = IntField(value="*@*", checker=int_check(lower=0), help="the output size")
        dropout = FloatField(value=0.0, checker=float_check(lower=0.0), help="the dropout rate")
        bias = BoolField(value=True, help="whether to use bias")
        pool = StrField(value=None, checker=str_check(options=['first'], additions=None), help="the pooling method, currently only implement `first`")

    config = NestField(value=Config, converter=nest_converter)


@register("module", "linear")
class Linear(Module):
    """wrap for nn.Linear"""
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self.config = config.config
        self.linear = nn.Linear(in_features=self.config.input_size, out_features=self.config.output_size)
        self.dropout = nn.Dropout(p=float(self.config.dropout))

    def forward(self, input: torch.Tensor)->torch.Tensor:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            project result the shape is the same as input(no poll), otherwise depend on poll method

        """
        output = self.dropout(self.linear(input))
        if not self.config.pool:
            return output
        elif self.config.pool == 'first':
            return output[:, 0]
        else:
            raise PermissionError(f"Currenttly we have not support the pool method '{self.config.pool}' in linear.")
