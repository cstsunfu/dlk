# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Dict, List, Optional

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
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder

from dlk.nn.module import Module
from dlk.utils.io import open
from dlk.utils.register import register


@cregister("module", "bart_encoder")
class BartEncoderWrapConfig(Base):
    pretrained_model_path = StrField(value="???", help="the pretrained model path")
    from_pretrain = BoolField(value=True, help="from pretrained or not")
    freeze = BoolField(value=False, help="freeze or not")


@register("module", "bart_encoder")
class BartEncoderWrap(Module):
    """BartEncoder wrap"""

    def __init__(self, config: BartEncoderWrapConfig, embedding: nn.Embedding = None):
        super(BartEncoderWrap, self).__init__()
        self.config = config
        if os.path.isdir(config.pretrained_model_path):
            if os.path.exists(
                os.path.join(config.pretrained_model_path, "config.json")
            ):
                with open(
                    os.path.join(config.pretrained_model_path, "config.json"), "r"
                ) as f:
                    self.bart_encoder_config = BartConfig(**json.load(f))
            else:
                raise PermissionError(
                    f"config.json must in the dir {self.pretrained_model_path}"
                )
        else:
            if os.path.isfile(config.pretrained_model_path):
                try:
                    with open(config.pretrained_model_path, "r") as f:
                        self.bart_encoder_config = BartConfig(**json.load(f))
                except:
                    raise PermissionError(
                        f"You must provide the pretrained model dir or the config file path."
                    )

        self.bart_encoder = BartEncoder(
            self.bart_encoder_config, embed_tokens=embedding
        )

    def init_weight(self, method):
        """init the weight of model by 'bart_encoder.init_weight()' or from_pretrain

        Args:
            method: init method, no use for pretrained_transformers

        Returns:
            None

        """
        if self.config.from_pretrain:
            self.from_pretrained()
        else:
            self.bart_encoder.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path"""
        self.bart_encoder: BartEncoder = BartEncoder.from_pretrained(
            self.config.pretrained_model_path
        )

    def forward(self, inputs: Dict):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            sequence_output, all_hidden_states, all_self_attentions

        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.bart_encoder(
                    input_ids=None,  # NOTE: we will add embedding in embedding layer
                    attention_mask=inputs.get("attention_mask", None),
                    head_mask=inputs.get("head_mask", None),
                    inputs_embeds=inputs.get("inputs_embeds", None),
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=False,
                )
        else:
            outputs = self.bart_encoder(
                input_ids=None,  # NOTE: we will add embedding in embedding layer
                attention_mask=inputs.get("attention_mask", None),
                head_mask=inputs.get("head_mask", None),
                inputs_embeds=inputs.get("inputs_embeds", None),
                output_attentions=True,
                output_hidden_states=True,
                return_dict=False,
            )
        assert (
            len(outputs) == 3
        ), f"Please check transformers version, the len(outputs) is 3 in version == 4.12|4.15"
        sequence_output, all_hidden_states, all_self_attentions = (
            outputs[0],
            outputs[1],
            outputs[2],
        )
        return sequence_output, all_hidden_states, all_self_attentions
