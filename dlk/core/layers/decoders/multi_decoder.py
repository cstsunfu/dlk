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
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from dlk.utils.config import BaseConfig, ConfigTool
from . import decoder_register, decoder_config_register
import torch.nn as nn

@decoder_config_register("multi_decoder")
class MultiDecoderConfig(BaseModuleConfig):
    default_config = {
        "config": {
            "module_rank": []
        },
        "_name": "multi_decoder",
    }
    """Config for MultiDecoder 
    """
    def __init__(self, config: Dict):
        super(MultiDecoderConfig, self).__init__(config)
        module_rank = config['config']['module_rank']
        self.decode_configs = {}
        self.module_rank = []
        for decode in module_rank:
            if decode not in config and "@" not in decode:
                decode = f"decoder@{decode}"
            if decode not in config:
                raise KeyError(f"{decode} not configured")
            self.module_rank.append(decode)
            module_class, module_config = ConfigTool.get_leaf_module(decoder_register, decoder_config_register, "decoder", config[decode])
            self.decode_configs[decode] = {
                "decode_class": module_class,
                "decode_config": module_config,
            }

        self.post_check(config, used=[
            "module_rank",
        ])


@decoder_register("multi_decoder")
class MultiDecoder(SimpleModule):
    """multi_decoder a x A x b
    """
    def __init__(self, config: MultiDecoderConfig):
        super(MultiDecoder, self).__init__(config)
        self._provide_keys = {'logits'}
        self._required_keys = {'embedding'}
        self._provided_keys = set()
        decode_configs = config.decode_configs
        self.decoders = nn.ModuleDict({
            decode_name: decode_configs[decode_name]['decode_class'](decode_configs[decode_name]['decode_config'])
            for decode_name in decode_configs
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
        for decode_name in self.config.module_rank:
            inputs = self.decoders[decode_name](inputs)
        return inputs
