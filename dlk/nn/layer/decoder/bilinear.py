# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Set

import torch
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
from dlk.nn.module.bilinear import BiLinear, BiLinearConfig
from dlk.utils.register import register


@cregister("decoder", "bilinear")
class DecoderBiLinearConfig(BiLinearConfig):
    """the bilinear decoder module"""

    class InputMap:
        attention_mask = StrField(value="attention_mask", help="the attention mask")
        embedding = StrField(value="embedding", help="the embedding")

    input_map = NestField(value=InputMap, help="the input map of the bilinear module")

    class OutputMap:
        logits = StrField(value="logits", help="the logits")

    output_map = NestField(
        value=OutputMap, help="the output map of the bilinear module"
    )
    apply_mask = BoolField(
        value=True,
        help="whether to apply padding mask and trail mask to the logits",
    )
    apply_tril_mask = BoolField(
        value=True,
        help="whether to apply trail mask to the logits",
    )


@register("decoder", "bilinear")
class DecoderBiLinear(SimpleModule):
    """bilinear"""

    def __init__(self, config: DecoderBiLinearConfig):
        super(DecoderBiLinear, self).__init__(config)
        self.config = config
        self.bilinear = BiLinear(config)

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.bilinear.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs
            Logits tensor, shape==(batch_size, output_size, seq_len, seq_len)
        """
        # logits:(batch_size, output_size, seq_len, seq_len)
        logits = self.bilinear(inputs[self.config.input_map.embedding])
        shape = logits.shape

        # padding mask
        if self.config.apply_mask:
            pad_mask = (
                inputs[self.config.input_map.attention_mask]
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(shape[0], shape[1], shape[2], shape[3])
            )
            logits = torch.where(pad_mask.to(torch.bool), logits, -torch.inf)
        if self.config.apply_tril_mask:
            tril_mask = torch.tril(torch.ones_like(logits), -1)
            logits = torch.where(tril_mask.to(torch.bool), -torch.inf, logits)

        inputs[self.config.output_map.logits] = logits
        return inputs
