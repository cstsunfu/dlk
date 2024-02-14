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
    dropout = FloatField(value=0.0, minimum=0.0, maximum=1.0, help="the dropout rate")


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
        if self.config.freeze:
            self.distil_bert.eval()
            with torch.no_grad():
                outputs = self.distil_bert(
                    input_ids=inputs.get("input_ids", None),
                    attention_mask=inputs.get("attention_mask", None),
                    head_mask=inputs.get("head_mask", None),
                    inputs_embeds=inputs.get("inputs_embeds", None),
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=False,
                )
        else:
            outputs = self.distil_bert(
                input_ids=inputs.get("input_ids", None),
                attention_mask=inputs.get("attention_mask", None),
                head_mask=inputs.get("head_mask", None),
                inputs_embeds=inputs.get("inputs_embeds", None),
                output_attentions=True,
                output_hidden_states=True,
                return_dict=False,
            )
        assert (
            len(outputs) == 3
        ), f"Please check transformers version, the len(outputs) is 3 for version == 4.12, and this version the output logistic of distil_bert is not as the same as bert and roberta."
        sequence_output, all_hidden_states, all_self_attentions = (
            outputs[0],
            outputs[1],
            outputs[2],
        )
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
