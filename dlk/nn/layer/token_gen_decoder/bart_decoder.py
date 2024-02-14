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
from dlk.nn.module.module_bart_decoder import BartDecoderWrap, BartDecoderWrapConfig
from dlk.utils.register import register


@cregister("token_gen_decoder", "bart_like_decoder")
class BartLikeDecoderConfig(BartDecoderWrapConfig):
    """the bart decoder module"""

    class InputMap:
        decoder_attention_mask = StrField(
            value="decoder_attention_mask", help="the decoder attention mask"
        )
        encoder_output_embedding = StrField(
            value="encoder_output_embedding", help="the encoder outputs"
        )
        decoder_head_mask = StrField(
            value="decoder_head_mask", help="the decoder head mask"
        )
        decoder_input_embedding = StrField(
            value="decoder_input_embedding", help="the input embedding"
        )

    bart_like_module_name = StrField(
        value="bart_decoder",
        help="the bart like decoder module name, like `bart_decoder` ",
    )
    input_map = NestField(value=InputMap, help="the input map of the linear module")

    class OutputMap:
        decoder_output_embedding = StrField(
            value="decoder_output_embedding", help="the output embedding"
        )
        decoder_past_cache = StrField(value="decoder_past_cache", help="the past cache")

    output_map = NestField(value=OutputMap, help="the output map of the linear module")


@register("token_gen_decoder", "bart_like_decoder")
class BartLikeDecoder(SimpleModule):
    """bart_like_decoder"""

    def __init__(self, config: BartLikeDecoderConfig, embedding: nn.Embedding):
        super(BartLikeDecoder, self).__init__(config)
        self.config = config

        self.bart_like_decoder = register.get("module", config.bart_like_module_name)(
            config=config, embedding=embedding
        )

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.bart_like_decoder.init_weight(method)

    def reorder_incremental_state(
        self,
        decoder_past_cache,
        new_order,
    ):
        if not decoder_past_cache:
            return decoder_past_cache
        return self.bart_like_decoder.reorder_incremental_state(
            decoder_past_cache, new_order
        )

    def forward(
        self, inputs: Dict[str, torch.Tensor], decoder_past_cache=None
    ) -> Dict[str, torch.Tensor]:
        """

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        module_inputs = {
            "decoder_attention_mask": inputs.get(
                self.config.input_map.decoder_attention_mask, None
            ),
            "encoder_outputs": inputs[self.config.input_map.encoder_output_embedding],
            "decoder_head_mask": inputs.get(
                self.config.input_map.decoder_head_mask, None
            ),
            "past_caches": decoder_past_cache,
            "inputs_embeds": inputs[self.config.input_map.decoder_input_embedding],
        }
        (
            hidden_states,
            next_cache,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
        ) = self.bart_like_decoder(module_inputs)
        inputs[self.config.output_map.decoder_output_embedding] = hidden_states
        # rewrite the cache
        inputs[self.config.output_map.decoder_past_cache] = next_cache
        return inputs
