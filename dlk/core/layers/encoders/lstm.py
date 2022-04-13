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
from . import encoder_register, encoder_config_register
from dlk.core.modules import module_config_register, module_register
from dlk.utils.logger import Logger
logger = Logger.get_logger()

@encoder_config_register("lstm")
class LSTMConfig(BaseModuleConfig):
    """Config for LSTM

    Config Example:
        >>> {
        >>>     module: {
        >>>         _base: "lstm",
        >>>     },
        >>>     config: {
        >>>         input_map: {},
        >>>         output_map: {},
        >>>         input_size: *@*,
        >>>         output_size: "*@*",
        >>>         num_layers: 1,
        >>>         dropout: "*@*", // dropout between layers
        >>>     },
        >>>     _link: {
        >>>         config.input_size: [module.config.input_size],
        >>>         config.output_size: [module.config.output_size],
        >>>         config.dropout: [module.config.dropout],
        >>>     },
        >>>     _name: "lstm",
        >>> }
    """

    def __init__(self, config: Dict):
        super(LSTMConfig, self).__init__(config)
        self.lstm_config = config["module"]
        assert self.lstm_config['_name'] == "lstm"
        self.post_check(config['config'], used=[
            "input_size",
            "output_size",
            "num_layers",
            "return_logits",
            "dropout",
        ])


@encoder_register("lstm")
class LSTM(SimpleModule):
    """Wrap for torch.nn.LSTM
    """
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__(config)
        self._provide_keys = {'embedding'}
        self._required_keys = {'embedding', 'attention_mask'}
        self._provided_keys = set()
        self.config = config
        self.lstm = module_register.get('lstm')(module_config_register.get('lstm')(config.lstm_config))

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
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('embedding')]]))
        return inputs
