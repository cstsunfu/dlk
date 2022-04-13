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
from . import encoder_register, encoder_config_register
from dlk.core.modules import module_config_register, module_register

@encoder_config_register("linear")
class LinearConfig(BaseModuleConfig):
    """Config for Linear 

    Config Example:
        >>> {
        >>>     "module": {
        >>>         "_base": "linear",
        >>>     },
        >>>     "config": {
        >>>         "input_size": "*@*",
        >>>         "output_size": "*@*",
        >>>         "pool": null,
        >>>         "dropout": 0.0,
        >>>         "output_map": {},
        >>>         "input_map": {}, // required_key: provide_key
        >>>     },
        >>>     "_link":{
        >>>         "config.input_size": ["module.config.input_size"],
        >>>         "config.output_size": ["module.config.output_size"],
        >>>         "config.pool": ["module.config.pool"],
        >>>     },
        >>>     "_name": "linear",
        >>> }
    """
    def __init__(self, config: Dict):
        super(LinearConfig, self).__init__(config)
        self.linear_config = config["module"]
        self.post_check(config['config'], used=[
            "input_size",
            "output_size",
            "return_logits",
            "pool",
            "dropout",
        ])


@encoder_register("linear")
class Linear(SimpleModule):
    """wrap for torch.nn.Linear
    """
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__(config)
        self._provide_keys = {'embedding'}
        self._required_keys = {'embedding'}
        self._provided_keys = set()

        self.config = config

        self.linear = module_register.get('linear')(module_config_register.get('linear')(config.linear_config))

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
        inputs[self.get_output_name("embedding")] = self.linear(inputs[self.get_input_name('embedding')])
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('embedding')]]))
        return inputs
