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
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from . import decoder_register, decoder_config_register
from dlk.core.modules import module_config_register, module_register
import torch.nn as nn

@decoder_config_register("biaffine")
class BiAffineConfig(BaseModuleConfig):
    """Config for BiAffine 

    Config Example:
        >>> {
        >>>     "module": {
        >>>         "_base": "biaffine",
        >>>     },
        >>>     "config": {
        >>>         "input_size": "*@*",
        >>>         "hidden_size": 0, //default equals to input_size
        >>>         "output_size": "*@*",
        >>>         "dropout": 0.0,
        >>>         "output_map": {},
        >>>         "input_map": {}, // required_key: provide_key
        >>>     },
        >>>     "_link":{
        >>>         "config.input_size": ["module.config.hidden_size"],
        >>>         "config.output_size": ["module.config.output_size"],
        >>>     },
        >>>     "_name": "biaffine",
        >>> }
    """
    def __init__(self, config: Dict):
        super(BiAffineConfig, self).__init__(config)
        self.biaffine_config = config["module"]
        config = config['config']
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        if not self.hidden_size:
            self.hidden_size = self.input_size
            self.biaffine_config['input_size'] = self.hidden_size
        self.dropout = config['dropout']
        self.post_check(config, used=[
            "input_size",
            "output_size",
            "pool",
            "dropout",
            "return_logits",
        ])


@decoder_register("biaffine")
class BiAffine(SimpleModule):
    """biaffine a x A x b
    """
    def __init__(self, config: BiAffineConfig):
        super(BiAffine, self).__init__(config)
        self._provide_keys = {'logits'}
        self._required_keys = {'embedding'}
        self._provided_keys = set()

        self.config = config
        self.linear_a = nn.Linear(config.input_size, config.hidden_size)
        self.linear_b = nn.Linear(config.input_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout)
        self.active = nn.LeakyReLU() # TODO: why GELU get loss nan?

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
        input_a = self.dropout(self.active(self.linear_a(embedding)))
        input_b = self.dropout(self.active(self.linear_b(embedding)))
        inputs[self.get_output_name("logits")] = self.biaffine(input_a, input_b)
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('logits')]]))
        return inputs
