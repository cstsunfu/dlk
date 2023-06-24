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

from transformers.models.auto import AutoModel, AutoConfig
import torch.nn as nn
import torch
from typing import Dict
from . import Module
from dlk.utils.io import open
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

@config_register("module", 'bert_like')
class BertLikeConfig(BaseConfig):
    name = NameField(value="bert_like", file=__file__, help="the bert like pretrained transformer config")
    @define
    class Config:
        pretrained_model_path = StrField(value="*@*", help="the pretrained model path")
        from_pretrain = BoolField(value=True, help="whether to load the pretrained model")
        freeze = BoolField(value=False, help="whether to freeze the model")
        dropout = FloatField(value=0.0, checker=float_check(lower=0.0), help="the dropout rate")
    config = NestField(value=Config, converter=nest_converter)


@config_register("module", 'bert_like')
class BertLike(Module):
    """docstring for TransformerEncoder"""
    def __init__(self, config: BertLikeConfig):
        super(BertLike, self).__init__()
        self.config = config.config
        self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_path)
        self.model = AutoModel.from_config(self.model_config)
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
            self.model.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path
        """
        self.model = AutoModel.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs: Dict):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            sequence_output, all_hidden_states, all_self_attentions

        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.model(
                    input_ids = inputs.get("input_ids", None),
                    attention_mask = inputs.get("attention_mask", None),
                    token_type_ids = inputs.get("token_type_ids", None),
                    position_ids = inputs.get("position_ids", None),
                    head_mask = inputs.get("head_mask", None),
                    use_cache = None,
                    output_attentions = True,
                    output_hidden_states = True,
                    return_dict = True
                )
        else:
            outputs = self.model(
                input_ids = inputs.get("input_ids", None),
                attention_mask = inputs.get("attention_mask", None),
                token_type_ids = inputs.get("token_type_ids", None),
                position_ids = inputs.get("position_ids", None),
                head_mask = inputs.get("head_mask", None),
                use_cache = None,
                output_attentions = True,
                output_hidden_states = True,
                return_dict = True
            )
        sequence_output, all_hidden_states, all_self_attentions = outputs.last_hidden_state, outputs.hidden_states, outputs.attentions
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
