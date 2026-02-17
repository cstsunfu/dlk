# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Dict

import torch
import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaModel

from dlk.nn.module.bert import BertWrap, BertWrapConfig
from dlk.utils.io import open
from dlk.utils.register import register

from . import Module


@cregister("module", "roberta")
class RobertaWrapConfig(BertWrapConfig):
    """the Roberta config"""

    pass


@register("module", "roberta")
class RobertaWrap(BertWrap):
    """Roberta Wrap"""

    def __init__(self, config: RobertaWrapConfig):
        super(RobertaWrap, self).__init__(config)
        self.config = config
        if os.path.isdir(self.config.pretrained_model_path):
            if os.path.exists(
                os.path.join(self.config.pretrained_model_path, "config.json")
            ):
                with open(
                    os.path.join(self.config.pretrained_model_path, "config.json"), "r"
                ) as f:
                    self.bert_config = RobertaConfig(**json.load(f))
            else:
                raise PermissionError(
                    f"config.json must in the dir {self.pretrained_model_path}"
                )
        else:
            if os.path.isfile(self.config.pretrained_model_path):
                try:
                    with open(self.config.pretrained_model_path, "r") as f:
                        self.bert_config = RobertaConfig(**json.load(f))
                except:
                    raise PermissionError(
                        f"You must provide the pretrained model dir or the config file path."
                    )

        self.roberta = RobertaModel(self.bert_config, add_pooling_layer=False)

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
        """init the model from pretrained_model_path"""
        self.roberta = RobertaModel.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            sequence_output, all_hidden_states, all_self_attentions

        """
        model_kwargs = {
            "input_ids": inputs.get("input_ids", None),
            "attention_mask": inputs.get("attention_mask", None),
            "token_type_ids": inputs.get("token_type_ids", None),
            "position_ids": inputs.get("position_ids", None),
            "head_mask": inputs.get("head_mask", None),
            "use_cache": None,
            "output_attentions": self.config.return_attention,
            "output_hidden_states": True,
            "return_dict": True,
        }

        if self.config.freeze:
            with torch.no_grad():
                outputs = self.roberta(**model_kwargs)
        else:
            outputs = self.roberta(**model_kwargs)

        sequence_output = outputs.last_hidden_state
        all_hidden_states = getattr(outputs, "hidden_states", None)
        all_self_attentions = getattr(outputs, "attentions", None)
        return sequence_output, all_hidden_states, all_self_attentions
