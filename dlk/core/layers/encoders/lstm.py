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
from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from dlk import register, config_register
from dlk.core.modules.lstm import LSTMConfig, LSTM
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter, ConfigTool
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

@config_register("encoder", 'lstm')
@define
class EncoderLSTMConfig(LSTMConfig):
    name = NameField(value="linear", file=__file__, help="the linear encoder module")
    @define
    class Config(LSTMConfig.Config):
        output_map = DictField(value={
            "embedding": "embedding"
            }, help="the output map of the biaffine module")
        input_map = DictField(value={
            "embedding": "embedding",
            "attention_mask": "attention_mask"
            }, help="the input map of the biaffine module")

    config = NestField(value=Config, converter=nest_converter)


@register("encoder", "lstm")
class EncoderLSTM(SimpleModule):
    """Wrap for torch.nn.LSTM
    """
    def __init__(self, config: EncoderLSTMConfig):
        super(EncoderLSTM, self).__init__(config)
        self.config = config
        self.lstm = LSTM(config)

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.lstm.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """All step do this

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        inputs[self.get_output_name('embedding')] = self.lstm(inputs[self.get_input_name('embedding')], inputs[self.get_input_name('attention_mask')])
        return inputs
