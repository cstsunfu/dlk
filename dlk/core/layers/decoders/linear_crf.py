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
from dlk.core.base_module import BaseModule, BaseModuleConfig
from . import decoder_register, decoder_config_register
from dlk.core.modules import module_config_register, module_register
import copy

@decoder_config_register("linear_crf")
class LinearCRFConfig(BaseModuleConfig):
    """Config for LinearCRF 

    Config Example:
        >>> {
        >>>     "module@linear": {
        >>>         "_base": "linear",
        >>>     },
        >>>     "module@crf": {
        >>>         "_base": "crf",
        >>>     },
        >>>     "config": {
        >>>         "input_size": "*@*",  // the linear input_size
        >>>         "output_size": "*@*", // the linear output_size
        >>>         "reduction": "mean", // crf reduction method
        >>>         "output_map": {}, //provide_key: output_key
        >>>         "input_map": {} // required_key: provide_key
        >>>     },
        >>>     "_link":{
        >>>         "config.input_size": ["module@linear.config.input_size"],
        >>>         "config.output_size": ["module@linear.config.output_size", "module@crf.config.output_size"],
        >>>         "config.reduction": ["module@crf.config.reduction"],
        >>>     }
        >>>     "_name": "linear_crf",
        >>> }
    """
    def __init__(self, config: Dict):
        super(LinearCRFConfig, self).__init__(config)
        self.linear_config = config["module@linear"]
        self.crf_config = config["module@crf"]
        self.post_check(config['config'], used=[
                            'input_size',
                            'output_size',
                            'reduction',
                            "return_logits",
                        ])


@decoder_register("linear_crf")
class LinearCRF(BaseModule):
    """use torch.nn.Linear get the emission probability and fit to CRF"""
    def __init__(self, config: LinearCRFConfig):
        super(LinearCRF, self).__init__(config)
        self._provide_keys = {'logits', "predict_seq_label"}
        self._required_keys = {'embedding', 'label_ids', 'attention_mask'}

        self.config = config
        self.linear = module_register.get('linear')(module_config_register.get('linear')(config.linear_config))
        self.crf = module_register.get('crf')(module_config_register.get('crf')(config.crf_config))

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.linear.init_weight(method)
        self.crf.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do predict, only get the predict labels

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self.predict_step(inputs)

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do predict, only get the predict labels

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        logits = self.linear(inputs[self.get_input_name('embedding')])
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([logits]))
        inputs[self.get_output_name("predict_seq_label")] = self.crf(logits, inputs[self.get_input_name('attention_mask')])
        return inputs

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do training step, get the crf loss

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        logits = self.linear(inputs[self.get_input_name('embedding')])
        loss = self.crf.training_step(logits, inputs[self.get_input_name('label_ids')], inputs[self.get_input_name('attention_mask')])
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([logits]))
        inputs[self.get_output_name('loss')] = loss
        return inputs

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do validation step, get the crf loss and the predict labels

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        logits = self.linear(inputs[self.get_input_name('embedding')])
        loss = self.crf.training_step(logits, inputs[self.get_input_name('label_ids')], inputs[self.get_input_name('attention_mask')])
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([logits]))
        inputs[self.get_output_name('loss')] = loss
        inputs[self.get_output_name("predict_seq_label")] = self.crf(logits, inputs[self.get_input_name('attention_mask')])
        return inputs
