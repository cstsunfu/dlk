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
from dlk.utils.config import BaseConfig
from . import Module
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

@config_register("module", 'logits_gather')
@define
class LogitsGatherConfig(BaseConfig):
    name = NameField(value="logits_gather", file=__file__, help="the logits_gather config")
    @define
    class Config:
        prefix = StrField(value="gather_logits_", help="the prefix of the output name")
        gather_layer = DictField(value={}, help="""
                                 the gather layer config, like
                                 "gather_layer": {
                                     "0": {
                                         "map": "3", # the 0th layer not do scale output to "gather_logits_3", "gather_logits_" is the output name prefix, the "3" is map name
                                         "scale": {} # don't scale
                                         },
                                     "1": {
                                         "map": "4",  # the 1th layer scale output dim from 1024 to 200 and the output named "gather_logits_3"
                                         "scale": {"1024":"200"},
                                         }
                                     },
                                 default is empty dict
                                 """)
    config = NestField(value=Config, converter=nest_converter)


@register("module", "logits_gather")
class LogitsGather(Module):
    """Gather the output logits decided by config
    """
    def __init__(self, config: LogitsGatherConfig):
        super(LogitsGather, self).__init__()
        self.config = config.config
        self.layers_scale = nn.ModuleDict()
        self.layer_map: Dict[str, str] = {}
        self.prefix = self.config.prefix
        for layer, layer_config in self.config.gather_layer.items():
            self.layer_map[str(layer)] = str(layer_config['map'])
            if layer_config.get("scale", {}):
                scale = layer_config['scale']
                assert len(scale) == 1
                for from_dim, to_dim in scale.items:
                    self.layers_scale[str(layer)] = nn.Linear(int(from_dim), int(to_dim))

        if not self.layer_map:
            self.pass_gather = True
        else:
            self.pass_gather = False


    def forward(self, input: List[torch.Tensor])->Dict[str, torch.Tensor]:
        """gather the needed input to dict

        Args:
            batch: a mini batch inputs

        Returns: 
            some elements to dict

        """
        result = torch.jit.annotate(Dict[str, torch.Tensor], {})

        if self.pass_gather:
            return result
        for layer, layer_suffix in self.layer_map.items():
            if layer in self.layers_scale:
                result[self.prefix+layer_suffix] = self.layers_scale[layer](input[int(layer)])
            else:
                result[self.prefix+layer_suffix] = input[int(layer)]

        return result
