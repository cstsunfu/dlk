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
from dlk.core.base_module import BaseModule

from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter, ConfigTool
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("decoder", 'linear_crf')
@define
class DecoderLinearCRFConfig(BaseConfig):
    name = NameField(value="linear_crf", file=__file__, help="the linear_crf decoder module")
    @define
    class Config:
        input_size = IntField(value="*@*", checker=int_check(lower=0), help="the input size")
        output_size = IntField(value="*@*", checker=int_check(lower=0), help="the output size")
        reduction = StrField(value="mean", checker=str_check(options=["none", "sum", "mean", "token_mean"]), help="the reduction method")

        output_map = DictField(value={
            "predict_seq_label": "predict_seq_label",
            "loss": "loss"
            }, help="the output map of the biaffine module")
        input_map = DictField(value={
            "embedding": "embedding",
            "label_ids": "label_ids",
            "attention_mask": "attention_mask"
            }, help="the input map of the biaffine module")

    config = NestField(value=Config, converter=nest_converter)
    submods = SubModules({ 
                          "module@linear": "linear",
                          "module@crf": "crf",
                          })
    link = DictField(value={
                "config.input_size": ["module@linear.config.input_size"],
                "config.output_size": ["module@linear.config.output_size", "module@crf.config.output_size"],
                "config.reduction": ["module@crf.config.reduction"],
                })


@register("decoder", "linear_crf")
class DecoderLinearCRF(BaseModule):
    """use torch.nn.Linear get the emission probability and fit to CRF"""
    def __init__(self, config: DecoderLinearCRFConfig):
        super(DecoderLinearCRF, self).__init__(config)

        config_dict = config.to_dict()
        self.linear = register.get("module", 'linear')(config_register.get("module", 'linear')(config_dict['module@linear']))
        self.crf = register.get("module", 'crf')(config_register.get("module", 'crf')(config_dict['module@crf']))

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
        inputs[self.get_output_name('loss')] = loss
        inputs[self.get_output_name("predict_seq_label")] = self.crf(logits, inputs[self.get_input_name('attention_mask')])
        return inputs
