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

from transformers.models.distilbert.modeling_distilbert import DistilBertModel
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
import json
import os
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict
from dlk.utils.logger import Logger
from . import Module
from dlk.utils.io import open
from dlk.utils.io import open
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules
logger = Logger.get_logger()

@config_register("module", 'distil_bert')
@define
class DistilBertWrapConfig(BaseConfig):
    name = NameField(value="distil_bert", file=__file__, help="the distil_bert config")
    @define
    class Config:
        pretrained_model_path = StrField(value="*@*", help="the pretrained model path")
        from_pretrain = BoolField(value=True, help="whether to load the pretrained model")
        freeze = BoolField(value=False, help="whether to freeze the model")
        dropout = FloatField(value=0.0, checker=float_check(lower=0.0), help="the dropout rate")
    config = NestField(value=Config, converter=nest_converter)



@register("module", "distil_bert")
class DistilBertWrap(Module):
    """DistillBertWrap"""
    def __init__(self, config: DistilBertWrapConfig):
        super(DistilBertWrap, self).__init__()
        self.config = config.config
        if os.path.isdir(self.config.pretrained_model_path):
            if os.path.exists(os.path.join(self.config.pretrained_model_path, 'config.json')):
                with open(os.path.join(self.config.pretrained_model_path, 'config.json'), 'r') as f:
                    self.bert_config = DistilBertConfig(**json.load(f))
            else:
                raise PermissionError(f"config.json must in the dir {self.pretrained_model_path}")
        else:
            if os.path.isfile(self.config.pretrained_model_path):
                try:
                    with open(self.config.pretrained_model_path, 'r') as f:
                        self.bert_config = DistilBertConfig(**json.load(f))
                except:
                    raise PermissionError(f"You must provide the pretrained model dir or the config file path.")

        self.distil_bert = DistilBertModel(self.bert_config)
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
            logger.info(f'Training the distill bert from scratch')
            self.distil_bert.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path
        """
        logger.info(f'Init the distill bert from {self.config.pretrained_model_path}')
        self.distil_bert = DistilBertModel.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            sequence_output, all_hidden_states, all_self_attentions

        """
        if self.config.freeze:
            self.distil_bert.eval()
            with torch.no_grad():
                outputs = self.distil_bert(
                    input_ids = inputs.get("input_ids", None),
                    attention_mask = inputs.get("attention_mask", None),
                    head_mask = inputs.get("head_mask", None),
                    inputs_embeds = inputs.get("inputs_embeds", None),
                    output_attentions = True,
                    output_hidden_states = True,
                    return_dict = False
                )
        else:
            outputs = self.distil_bert(
                input_ids = inputs.get("input_ids", None),
                attention_mask = inputs.get("attention_mask", None),
                head_mask = inputs.get("head_mask", None),
                inputs_embeds = inputs.get("inputs_embeds", None),
                output_attentions = True,
                output_hidden_states = True,
                return_dict = False
            )
        assert len(outputs) == 3, f"Please check transformers version, the len(outputs) is 3 for version == 4.12, and this version the output logistic of distil_bert is not as the same as bert and roberta."
        sequence_output, all_hidden_states, all_self_attentions = outputs[0], outputs[1], outputs[2]
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
