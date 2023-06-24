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

from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
import json
import torch
import os
import torch.nn as nn
from typing import Dict
from . import Module
from dlk.utils.io import open
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules
from dlk.core.modules.bert import BertWrapConfig, BertWrap

@config_register("module", 'roberta')
@define
class RobertaWrapConfig(BertWrapConfig):
    name = NameField(value="roberta", file=__file__, help="the roberta config")


@register("module", "roberta")
class RobertaWrap(BertWrap):
    """Roberta Wrap"""
    def __init__(self, config: RobertaWrapConfig):
        super(RobertaWrap, self).__init__(config)
        self.config = config.config
        if os.path.isdir(self.config.pretrained_model_path):
            if os.path.exists(os.path.join(self.config.pretrained_model_path, 'config.json')):
                with open(os.path.join(self.config.pretrained_model_path, 'config.json'), 'r') as f:
                    self.bert_config = RobertaConfig(**json.load(f))
            else:
                raise PermissionError(f"config.json must in the dir {self.pretrained_model_path}")
        else:
            if os.path.isfile(self.config.pretrained_model_path):
                try:
                    with open(self.config.pretrained_model_path, 'r') as f:
                        self.bert_config = RobertaConfig(**json.load(f))
                except:
                    raise PermissionError(f"You must provide the pretrained model dir or the config file path.")

        self.roberta = RobertaModel(self.bert_config, add_pooling_layer=False)
        self.dropout = nn.Dropout(float(self.config.dropout))

    def init_weight(self, method):
        """init the weight of model by 'bert.init_weight()' or from_pretrain

        Args:
            method: init method, no use for pretrained_transformers

        Returns: 
            None

        """
        if self.config.from_pretrain:
            self.from_pretrained()
        else:
            self.roberta.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path
        """
        self.roberta = RobertaModel.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            sequence_output, all_hidden_states, all_self_attentions

        """
        if self.config.freeze:
            self.roberta.eval()
            with torch.no_grad():
                outputs = self.roberta(
                        input_ids = inputs.get("input_ids", None),
                        attention_mask = inputs.get("attention_mask", None),
                        token_type_ids = inputs.get("token_type_ids", None),
                        position_ids = inputs.get("position_ids", None),
                        head_mask = inputs.get("head_mask", None),
                        inputs_embeds = inputs.get("inputs_embeds", None),
                        encoder_hidden_states = inputs.get("encoder_hidden_states", None),
                        encoder_attention_mask = inputs.get("encoder_attention_mask", None),
                        past_key_values = inputs.get("past_key_values", None),
                        use_cache = None,
                        output_attentions = True,
                        output_hidden_states = True,
                        return_dict = False
                        )
        else:
            outputs = self.roberta(
                    input_ids = inputs.get("input_ids", None),
                    attention_mask = inputs.get("attention_mask", None),
                    token_type_ids = inputs.get("token_type_ids", None),
                    position_ids = inputs.get("position_ids", None),
                    head_mask = inputs.get("head_mask", None),
                    inputs_embeds = inputs.get("inputs_embeds", None),
                    encoder_hidden_states = inputs.get("encoder_hidden_states", None),
                    encoder_attention_mask = inputs.get("encoder_attention_mask", None),
                    past_key_values = inputs.get("past_key_values", None),
                    use_cache = None,
                    output_attentions = True,
                    output_hidden_states = True,
                    return_dict = False
                    )
        assert len(outputs) == 4, f"Please check transformers version, the len(outputs) is 4 in version == 4.12, or check your config and remove the 'add_cross_attention'"
        sequence_output, all_hidden_states, all_self_attentions = outputs[0], outputs[2], outputs[3]
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
