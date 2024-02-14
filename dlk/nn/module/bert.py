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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel

from dlk.utils.io import open
from dlk.utils.register import register

from . import Module


@cregister("module", "bert")
class BertWrapConfig:
    """the BERT config"""

    pretrained_model_path = StrField(value=MISSING, help="the pretrained model path")
    from_pretrain = BoolField(value=True, help="whether to load the pretrained model")
    freeze = BoolField(value=False, help="whether to freeze the model")
    dropout = FloatField(value=0.0, minimum=0.0, maximum=1.0, help="the dropout rate")


@register("module", "bert")
class BertWrap(Module):
    """Bert wrap"""

    def __init__(self, config: BertWrapConfig):
        super(BertWrap, self).__init__()
        self.config = config
        if os.path.isdir(self.config.pretrained_model_path):
            if os.path.exists(
                os.path.join(self.config.pretrained_model_path, "config.json")
            ):
                with open(
                    os.path.join(self.config.pretrained_model_path, "config.json"), "r"
                ) as f:
                    self.bert_config = BertConfig(**json.load(f))
            else:
                raise PermissionError(
                    f"config.json must in the dir {self.pretrained_model_path}"
                )
        else:
            if os.path.isfile(self.config.pretrained_model_path):
                try:
                    with open(self.config.pretrained_model_path, "r") as f:
                        self.bert_config = BertConfig(**json.load(f))
                except:
                    raise PermissionError(
                        f"You must provide the pretrained model dir or the config file path."
                    )

        self.bert = BertModel(self.bert_config, add_pooling_layer=True)
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
            self.bert.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path"""
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs: Dict):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            sequence_output, all_hidden_states, all_self_attentions

        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=inputs.get("input_ids", None),
                    attention_mask=inputs.get("attention_mask", None),
                    token_type_ids=inputs.get("token_type_ids", None),
                    position_ids=inputs.get("position_ids", None),
                    head_mask=inputs.get("head_mask", None),
                    inputs_embeds=inputs.get("inputs_embeds", None),
                    encoder_hidden_states=inputs.get("encoder_hidden_states", None),
                    encoder_attention_mask=inputs.get("encoder_attention_mask", None),
                    past_key_values=inputs.get("past_key_values", None),
                    use_cache=None,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=False,
                )
        else:
            outputs = self.bert(
                input_ids=inputs.get("input_ids", None),
                attention_mask=inputs.get("attention_mask", None),
                token_type_ids=inputs.get("token_type_ids", None),
                position_ids=inputs.get("position_ids", None),
                head_mask=inputs.get("head_mask", None),
                inputs_embeds=inputs.get("inputs_embeds", None),
                encoder_hidden_states=inputs.get("encoder_hidden_states", None),
                encoder_attention_mask=inputs.get("encoder_attention_mask", None),
                past_key_values=inputs.get("past_key_values", None),
                use_cache=None,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=False,
            )
        assert (
            len(outputs) == 4
        ), f"Please check transformers version, the len(outputs) is 4 in version == 4.12|4.15, or check your config and remove the 'add_cross_attention'"
        sequence_output, all_hidden_states, all_self_attentions = (
            outputs[0],
            outputs[2],
            outputs[3],
        )
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
