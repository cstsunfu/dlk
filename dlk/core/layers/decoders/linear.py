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

@decoder_config_register("linear")
class LinearConfig(BaseModuleConfig):
    default_config = {
            "_name": "linear",
            "module": {
                "_base": "linear",
                },
            }
    """Config for Linear 

    Config Example:
        default_config
    """
    def __init__(self, config: Dict):
        super(LinearConfig, self).__init__(config)
        self.linear_config = config["module"]
        self.post_check(config['config'], used=[
            "input_size",
            "output_size",
            "pool",
            "dropout",
            "return_logits",
        ])


@decoder_register("linear")
class Linear(SimpleModule):
    """wrap for torch.nn.Linear
    """
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__(config)
        self._provide_keys = {'logits'}
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
        inputs[self.get_output_name("logits")] = self.linear(inputs[self.get_input_name('embedding')])
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('logits')]]))
        return inputs
