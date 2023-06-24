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

import torch
from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule
from dlk import register, config_register
from dlk.core.modules.linear import LinearConfig, Linear
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter, ConfigTool
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("decoder", 'linear')
@define
class DecoderLinearConfig(LinearConfig):
    name = NameField(value="linear", file=__file__, help="the linear decoder module")
    @define
    class Config(LinearConfig.Config):
        output_map = DictField(value={
            "logits": "logits"
            }, help="the output map of the biaffine module")
        input_map = DictField(value={
            "embedding": "embedding"
            }, help="the input map of the biaffine module")

    config = NestField(value=Config, converter=nest_converter)


@register("decoder", "linear")
class DecoderLinear(SimpleModule):
    """wrap for torch.nn.Linear
    """
    def __init__(self, config: DecoderLinearConfig):
        super(DecoderLinear, self).__init__(config)
        self.linear = register.get("module", 'linear')(config)

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.linear.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """All step do this

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        inputs[self.get_output_name("logits")] = self.linear(inputs[self.get_input_name('embedding')])
        return inputs
