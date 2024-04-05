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
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from dlk.nn.module import Module
from dlk.nn.module.module_bart_decoder import BartDecoderWrapConfig
from dlk.utils.io import open
from dlk.utils.register import register


@cregister("module", "gpt2_decoder")
class GPT2DecoderWrapConfig(BartDecoderWrapConfig):
    """the gpt2 decoder config, reuse the bart like decoder config"""

    pass


@register("module", "gpt2_decoder")
class GPT2DecoderWrap(Module):
    """GPT2Decoder wrap"""

    def __init__(self, config: GPT2DecoderWrapConfig, embedding: nn.Embedding):
        super(GPT2DecoderWrap, self).__init__()
        self.config = config
        if os.path.isdir(config.pretrained_model_path):
            if os.path.exists(
                os.path.join(config.pretrained_model_path, "config.json")
            ):
                with open(
                    os.path.join(config.pretrained_model_path, "config.json"), "r"
                ) as f:
                    self.gpt2_decoder_config = GPT2Config(**json.load(f))
            else:
                raise PermissionError(
                    f"config.json must in the dir {self.pretrained_model_path}"
                )
        else:
            if os.path.isfile(config.pretrained_model_path):
                try:
                    with open(config.pretrained_model_path, "r") as f:
                        self.gpt2_decoder_config = GPT2Config(**json.load(f))
                except:
                    raise PermissionError(
                        f"You must provide the pretrained model dir or the config file path."
                    )
            else:
                raise PermissionError(
                    f"Can not init the gpt2 decoder from {config.pretrained_model_path}."
                )

        self.gpt2_decoder = GPT2Model(self.gpt2_decoder_config)

    def init_weight(self, method):
        """init the weight of model by 'gpt2_decoder.init_weight()' or from_pretrain

        Args:
            method: init method, no use for pretrained_transformers

        Returns:
            None

        """
        if self.config.from_pretrain:
            self.from_pretrained()
        else:
            self.gpt2_decoder.init_weights()

    def from_pretrained(self):
        """init the model from pretrained_model_path"""
        self.gpt2_decoder: GPT2Model = GPT2Model.from_pretrained(
            self.config.pretrained_model_path
        )

    def reorder_incremental_state(
        self,
        past_key_values,
        beam_idx,
    ):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def forward(self, inputs: Dict):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            sequence_output, all_hidden_states, all_self_attentions

        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.gpt2_decoder(
                    input_ids=None,  # NOTE: we will add embedding in embedding layer
                    attention_mask=inputs.get("decoder_attention_mask", None),
                    encoder_hidden_states=inputs["encoder_outputs"],
                    head_mask=inputs.get("decoder_head_mask", None),
                    past_key_values=inputs.get("past_caches", None),
                    inputs_embeds=inputs["inputs_embeds"],
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=False,
                )
        else:
            outputs = self.gpt2_decoder(
                input_ids=None,  # NOTE: we will add embedding in embedding layer
                attention_mask=inputs.get("decoder_attention_mask", None),
                encoder_hidden_states=inputs["encoder_outputs"],
                head_mask=inputs.get("decoder_head_mask", None),
                past_key_values=inputs.get("past_caches", None),
                inputs_embeds=inputs["inputs_embeds"],
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=False,
            )
        assert (
            len(outputs) == 5
        ), f"Please check transformers version, the len(outputs) is 3 in version == 4.12|4.15"
        # sequence_output, all_hidden_states, all_self_attentions = outputs[0], outputs[1], outputs[2]
        (
            hidden_states,
            next_cache,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
        ) = (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        return (
            hidden_states,
            next_cache,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
        )
