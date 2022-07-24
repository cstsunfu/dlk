# Copyright cstsunfu. All rights reserved.
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

from transformers.models.bart.modeling_bart import BartDecoder
from transformers.models.bart.configuration_bart import BartConfig
import json
import os
import torch.nn as nn
import torch
from typing import Dict
from . import module_register, module_config_register, Module
from dlk.utils.config import BaseConfig
from dlk.utils.io import open


@module_config_register("bart_decoder")
class BartDecoderWrapConfig(BaseConfig):
    """Config for BartDecoderWrap

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "pretrained_model_path": "*@*",
        >>>         "from_pretrain": true,
        >>>         "freeze": false,
        >>>         "dropout": 0.0,
        >>>     },
        >>>     "_name": "bart_decoder",
        >>> }
    """

    def __init__(self, config: Dict):
        super(BartDecoderWrapConfig, self).__init__(config)
        self.pretrained_model_path = config['config']['pretrained_model_path']
        self.from_pretrain = config['config']['from_pretrain']
        self.freeze = config['config']['freeze']
        self.dropout = config['config']['dropout']
        if os.path.isdir(self.pretrained_model_path):
            if os.path.exists(os.path.join(self.pretrained_model_path, 'config.json')):
                with open(os.path.join(self.pretrained_model_path, 'config.json'), 'r') as f:
                    self.bart_decoder_config = BartConfig(**json.load(f))
            else:
                raise PermissionError(f"config.json must in the dir {self.pretrained_model_path}")
        else:
            if os.path.isfile(self.pretrained_model_path):
                try:
                    with open(self.pretrained_model_path, 'r') as f:
                        self.bart_decoder_config = BartConfig(**json.load(f))
                except:
                    raise PermissionError(f"You must provide the pretrained model dir or the config file path.")
        self.post_check(config['config'], used=['pretrained_model_path', 'from_pretrain', 'freeze', 'dropout'])


@module_register("bart_decoder")
class BartDecoderWrap(Module):
    """BartDecoder wrap"""
    def __init__(self, config: BartDecoderWrapConfig):
        super(BartDecoderWrap, self).__init__()
        self.config = config

        self.bart_decoder = BartDecoder(config.bart_decoder_config, embed_tokens=nn.Embedding(0, 0)) # NOTE: we will add embedding in embedding layer
        self.dropout = nn.Dropout(float(self.config.dropout))

    def init_weight(self, method):
        """init the weight of model by 'bart_decoder.init_weight()' or from_pretrain

        Args:
            method: init method, no use for pretrained_transformers

        Returns: 
            None

        """
        if self.config.from_pretrain:
            self.from_pretrained()
        else:
            self.bart_decoder.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path
        """
        self.bart_decoder: BartDecoder = BartDecoder.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs: Dict):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            sequence_output, all_hidden_states, all_self_attentions

        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.bart_decoder(
                    input_ids = None, # NOTE: we will add embedding in embedding layer
                    attention_mask = inputs.get("attention_mask", None),
                    head_mask = inputs.get("head_mask", None),
                    inputs_embeds = inputs.get("inputs_embeds", None),
                    output_attentions = True,
                    output_hidden_states = True,
                    return_dict = False
                )
        else:
            outputs = self.bart_decoder(
                input_ids = None, # NOTE: we will add embedding in embedding layer
                attention_mask = inputs.get("attention_mask", None),
                head_mask = inputs.get("head_mask", None),
                inputs_embeds = inputs.get("inputs_embeds", None),
                output_attentions = True,
                output_hidden_states = True,
                return_dict = False
            )
        assert len(outputs) == 3, f"Please check transformers version, the len(outputs) is 3 in version == 4.12|4.15"
        sequence_output, all_hidden_states, all_self_attentions = outputs[0], outputs[1], outputs[2]
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
