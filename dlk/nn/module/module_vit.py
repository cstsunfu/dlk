# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

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
from transformers import ViTModel
from transformers.models.auto import AutoConfig, AutoModel

from dlk.nn.module import Module
from dlk.utils.register import register


@cregister("module", "vit")
class VitConfig:
    """the vit pretrained config"""

    pretrained_model_path = StrField(value=MISSING, help="the pretrained model path")
    from_pretrain = BoolField(value=True, help="whether to load the pretrained model")
    freeze = BoolField(value=False, help="whether to freeze the model")
    dropout = FloatField(value=0.0, minimum=0.0, maximum=1.0, help="the dropout rate")


@register("module", "vit")
class Vit(Module):
    """docstring for vit"""

    def __init__(self, config: VitConfig):
        super(Vit, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(
            self.config.pretrained_model_path
        )
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
        """init the model from pretrained_model_path"""
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
                    pixel_values=inputs.get("pixel_values", None),
                    return_dict=True,
                )
        else:
            outputs = self.model(
                pixel_values=inputs.get("pixel_values", None),
                return_dict=True,
            )
        sequence_output, all_hidden_states, all_self_attentions = (
            outputs.last_hidden_state,
            outputs.hidden_states,
            outputs.attentions,
        )
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
