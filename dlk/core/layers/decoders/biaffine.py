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
from . import decoder_register, decoder_config_register
from dlk.core.modules import module_config_register, module_register
import torch.nn as nn

@decoder_config_register("biaffine")
class BiAffineConfig(BaseModuleConfig):
    default_config = {
        "_name": "biaffine",
        "config": {
            "input_size": "*@*",
            "hidden_size": 0, # default equals to input_size
            "output_size": "*@*",
            "dropout": 0.0,
            "max_seq_len": 1024,
            "relation_position": False,
            "multi_matrix": 1,
            "output_map": {},
            "input_map": {}, # required_key: provide_key
        },
        "_link":{
            "config.input_size": ["module.config.input_size"],
            "config.hidden_size": ["module.config.hidden_size"],
            "config.output_size": ["module.config.output_size"],
            "config.max_seq_len": ["module.config.max_seq_len"],
            "config.relation_position": ["module.config.relation_position"],
            "config.multi_matrix": ["module.config.multi_matrix"],
            "config.dropout": ["module.config.dropout"],
        },
        "module": {
            "_base": "biaffine",
        },
    }
    """Config for BiAffine 
    """
    def __init__(self, config: Dict):
        super(BiAffineConfig, self).__init__(config)
        self.biaffine_config = config["module"]
        config = config['config']
        self.post_check(config, used=[
            "input_size",
            "output_size",
            "hidden_size",
            "multi_matrix",
            "dropout",
            "max_seq_len",
            "relation_position",
            "return_logits",
        ])


@decoder_register("biaffine")
class BiAffine(SimpleModule):
    """biaffine
    """
    def __init__(self, config: BiAffineConfig):
        super(BiAffine, self).__init__(config)
        self._provide_keys = {'logits'}
        self._required_keys = {'embedding'}
        self._provided_keys = set()

        self.config = config

        self.biaffine = module_register.get('biaffine')(module_config_register.get('biaffine')(config.biaffine_config))

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.biaffine.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        embedding = inputs[self.get_input_name('embedding')]
        inputs[self.get_output_name("logits")] = self.biaffine(embedding)
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('logits')]]))
        return inputs
