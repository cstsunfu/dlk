# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Callable, Dict, List, Optional, Set

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

from dlk.nn.base_module import SimpleModule
from dlk.nn.module.module_bart_encoder import BartEncoderWrap, BartEncoderWrapConfig
from dlk.utils.register import register


@cregister("encoder", "bart_like_encoder")
class BartLikeEncoderConfig(BartEncoderWrapConfig):
    """the bart encoder module"""

    class InputMap:
        encoder_attention_mask = StrField(
            value="encoder_attention_mask", help="the decoder attention mask"
        )
        encoder_head_mask = StrField(
            value="encoder_head_mask", help="the decoder head mask"
        )
        encoder_input_embedding = StrField(
            value="encoder_input_embedding", help="the input embedding"
        )

    input_map = NestField(value=InputMap, help="the input map of the linear module")
    bart_like_module_name = StrField(
        value="bart_encoder",
        help="the bart like decoder module name",
    )

    class OutputMap:
        encoder_output_embedding = StrField(
            value="encoder_output_embedding", help="the encoder outputs"
        )

    output_map = NestField(value=OutputMap, help="the output map of the linear module")


@register("encoder", "bart_like_encoder")
class BartLikeEncoder(SimpleModule):
    """Wrap for torch.nn.BartEncoder"""

    def __init__(self, config: BartLikeEncoderConfig, embedding: nn.Embedding = None):
        super(BartLikeEncoder, self).__init__(config)
        self.config = config
        self.bart_like_encoder = register.get("module", config.bart_like_module_name)(
            config=config, embedding=embedding
        )

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.bart_like_encoder.init_weight(method)

    def reorder_encoder_out(self, encoder_outs: Dict[str, torch.Tensor], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        encoder_output_embedding = encoder_outs.get(
            self.config.output_map.encoder_output_embedding, None
        )
        if encoder_output_embedding is not None:
            encoder_outs[
                self.config.output_map.encoder_output_embedding
            ] = encoder_output_embedding.index_select(0, new_order)

        encoder_head_mask = encoder_outs.get(
            self.config.input_map.encoder_head_mask, None
        )
        if encoder_head_mask is not None:
            encoder_outs[
                self.config.input_map.encoder_head_mask
            ] = encoder_head_mask.index_select(0, new_order)

        encoder_input_embedding = encoder_outs.get(
            self.config.input_map.encoder_input_embedding, None
        )
        if encoder_input_embedding is not None:
            encoder_outs[
                self.config.input_map.encoder_input_embedding
            ] = encoder_input_embedding.index_select(0, new_order)

        return encoder_outs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """All step do this

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        module_inputs = {
            "attention_mask": inputs.get(
                self.config.input_map.encoder_attention_mask, None
            ),
            "head_mask": inputs.get(self.config.input_map.encoder_head_mask, None),
            "inputs_embeds": inputs.get(
                self.config.input_map.encoder_input_embedding, None
            ),
        }

        (
            sequence_output,
            all_hidden_states,
            all_self_attentions,
        ) = self.bart_like_encoder(module_inputs)
        inputs[self.config.output_map.encoder_output_embedding] = sequence_output
        return inputs
