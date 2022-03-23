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
from . import module_register, module_config_register, Module


@module_config_register("logits_gather")
class LogitsGatherConfig(BaseConfig):
    """Config for LogitsGather

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "gather_layer": {
        >>>             "0": {
        >>>                 "map": "3", // the 0th layer not do scale output to "gather_logits_3", "gather_logits_" is the output name prefix, the "3" is map name
        >>>                 "scale": {} //don't scale
        >>>             },
        >>>             "1": {
        >>>                 "map": "4",  // the 1th layer scale output dim from 1024 to 200 and the output named "gather_logits_3"
        >>>                 "scale": {"1024":"200"},
        >>>             }
        >>>         },
        >>>         "prefix": "gather_logits_",
        >>>     },
        >>>     _name: "logits_gather",
        >>> }
    """
    def __init__(self, config: Dict):
        if '_name' not in config:
            config['_name'] = 'logits_gather'
        super(LogitsGatherConfig, self).__init__(config)
        config = config.get('config', {})
        self.gather_layer = config.get('gather_layer', {})
        self.prefix = config.get("prefix", '')
        self.post_check(config, used=[
            "gather_layer",
            "prefix"
        ])


@module_register("logits_gather")
class LogitsGather(Module):
    """Gather the output logits decided by config
    """
    def __init__(self, config: LogitsGatherConfig):
        super(LogitsGather, self).__init__()
        gather_layer_num = len(config.gather_layer)
        self.layers_scale = nn.ModuleDict()
        self.layer_map: Dict[str, str] = {}
        self.prefix = config.prefix
        for layer, layer_config in config.gather_layer.items():
            self.layer_map[str(layer)] = str(layer_config['map'])
            if layer_config.get("scale", {}):
                scale = layer_config['scale']
                assert len(scale) == 1
                for from_dim, to_dim in scale.items:
                    self.layers_scale[str(layer)] = nn.Linear(int(from_dim), int(to_dim))

    def forward(self, input: List[torch.Tensor])->Dict[str, torch.Tensor]:
        """gather the needed input to dict

        Args:
            batch: a mini batch inputs

        Returns: 
            some elements to dict

        """
        result = torch.jit.annotate(Dict[str, torch.Tensor], {})

        if not self.layer_map:
            return result
        for layer, layer_suffix in self.layer_map.items():
            if layer in self.layers_scale:
                result[self.prefix+layer_suffix] = self.layers_scale[layer](input[int(layer)])
            else:
                result[self.prefix+layer_suffix] = input[int(layer)]

        return result
