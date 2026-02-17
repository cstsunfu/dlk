# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import DistilBertModel

from dlk.utils.io import open
from dlk.utils.register import register

from . import Module

logger = logging.getLogger(__name__)


@cregister("module", "distil_bert")
class DistilBertWrapConfig:
    """the distil_bert config"""

    pretrained_model_path = StrField(value=MISSING, help="the pretrained model path")
    from_pretrain = BoolField(value=True, help="whether to load the pretrained model")
    freeze = BoolField(value=False, help="whether to freeze the model")
    return_attention = BoolField(
        value=False,
        help="whether to return the attention weights, BertSdpaSelfAttention does not support this",
    )


@register("module", "distil_bert")
class DistilBertWrap(Module):
    """DistillBertWrap"""

    def __init__(self, config: DistilBertWrapConfig):
        super(DistilBertWrap, self).__init__()
        self.config = config
        if os.path.isdir(self.config.pretrained_model_path):
            if os.path.exists(
                os.path.join(self.config.pretrained_model_path, "config.json")
            ):
                with open(
                    os.path.join(self.config.pretrained_model_path, "config.json"), "r"
                ) as f:
                    self.bert_config = DistilBertConfig(**json.load(f))
            else:
                raise PermissionError(
                    f"config.json must in the dir {self.pretrained_model_path}"
                )
        else:
            if os.path.isfile(self.config.pretrained_model_path):
                try:
                    with open(self.config.pretrained_model_path, "r") as f:
                        self.bert_config = DistilBertConfig(**json.load(f))
                except:
                    raise PermissionError(
                        f"You must provide the pretrained model dir or the config file path."
                    )

        self.distil_bert = DistilBertModel(self.bert_config)

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
            logger.info(f"Training the distill bert from scratch")
            self.distil_bert.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path"""
        logger.info(f"Init the distill bert from {self.config.pretrained_model_path}")
        self.distil_bert = DistilBertModel.from_pretrained(
            self.config.pretrained_model_path
        )

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
            "head_mask": inputs.get("head_mask", None),
            "output_attentions": self.config.return_attention,
            "output_hidden_states": True,
            "return_dict": True,  # Force returning dictionary
        }

        if self.config.freeze:
            with torch.no_grad():
                outputs = self.distil_bert(**model_kwargs)
        else:
            outputs = self.distil_bert(**model_kwargs)

        sequence_output, all_hidden_states, all_self_attentions = (
            outputs.last_hidden_state,
            outputs.hidden_states,
            outputs.attentions,
        )
        return sequence_output, all_hidden_states, all_self_attentions
