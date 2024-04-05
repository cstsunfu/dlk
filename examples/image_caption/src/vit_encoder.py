# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Set

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
from dlk.nn.module.module_vit import Vit, VitConfig
from dlk.utils.register import register


@cregister("encoder", "vit")
class VitEncoderConfig(VitConfig):
    """
    the bert like encoder module
    """

    encoder_dim = IntField(value=MISSING, minimum=1, help="the encoder dim")

    class InputMap:
        pixel_values = StrField(value="pixel_values", help="the input ids")
        encoder_attention_mask = StrField(
            value="encoder_attention_mask", help="the decoder attention mask"
        )
        encoder_head_mask = StrField(
            value="encoder_head_mask", help="the decoder head mask"
        )

    input_map = NestField(
        value=InputMap, help="the input map of the static encoder module"
    )

    class OutputMap:
        encoder_output_embedding = StrField(
            value="encoder_output_embedding", help="the encoder outputs"
        )

    output_map = NestField(
        value=OutputMap, help="the output map of the static encoder module"
    )


@register("encoder", "vit")
class VitEncoder(SimpleModule):
    """Wrap the hugingface transformers"""

    def __init__(self, config: VitEncoderConfig, **args):
        super(VitEncoder, self).__init__(config)
        self.config = config
        self.pretrained_transformers = Vit(config)

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.pretrained_transformers.init_weight(method)

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
            self.config.input_map.pixel_values, None
        )
        if encoder_input_embedding is not None:
            encoder_outs[
                self.config.input_map.pixel_values
            ] = encoder_input_embedding.index_select(0, new_order)

        return encoder_outs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """get the transformers output as encoder

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        pixel_values = inputs[self.config.input_map.pixel_values]
        (
            sequence_output,
            all_hidden_states,
            all_self_attentions,
        ) = self.pretrained_transformers({"pixel_values": pixel_values})
        inputs[self.config.output_map.encoder_output_embedding] = sequence_output
        return inputs
