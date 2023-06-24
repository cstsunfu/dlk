# Copyright cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule
import torch.nn as nn
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter, ConfigTool
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("decoder", 'multi_decoder')
@define
class MultiDecoderConfig(BaseConfig):
    name = NameField(value="multi_decoder", file=__file__, help="the multi_decoder module")
    config = DictField(value={}, help='the config of the multi_decoder module')
    submods = SubModules({
    }, help="the modules config")


@register("decoder", "multi_decoder")
class MultiDecoder(SimpleModule):
    """multi_decoder a x A x b
    """
    def __init__(self, config: MultiDecoderConfig):
        super(MultiDecoder, self).__init__(config)
        config_dict = config.to_dict()
        self.decode_configs = {}
        for decode in config_dict:
            if decode in {"config", "name", "base"}:
                continue
            module_class, module_config = ConfigTool.get_leaf_module(register, config_register, "decoder", config_dict[decode])
            self.decode_configs[decode] = {
                "decode_class": module_class,
                "decode_config": module_config,
            }
        self.decoders = nn.ModuleDict({
            decode_name: self.decode_configs[decode_name]['decode_class'](self.decode_configs[decode_name]['decode_config'])
            for decode_name in self.decode_configs
        })

        self.config = config

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        for decode_name in self.decoders:
            self.decoders[decode_name].init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        for decode_name in self.decode_configs:
            inputs = self.decoders[decode_name](inputs)
        return inputs
