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
from dlk.nn.module.biaffine import BiAffine, BiAffineConfig
from dlk.utils.register import register


@cregister("decoder", "biaffine")
class DecoderBiAffineConfig(BiAffineConfig):
    """the biaffine decoder module"""

    class InputMap:
        embedding = StrField(value="embedding", help="the embedding")

    input_map = NestField(value=InputMap, help="the input map of the biaffine module")

    class OutputMap:
        logits = StrField(value="logits", help="the logits")

    output_map = NestField(
        value=OutputMap, help="the output map of the biaffine module"
    )


@register("decoder", "biaffine")
class DecoderBiAffine(SimpleModule):
    """biaffine"""

    def __init__(self, config: DecoderBiAffineConfig):
        super(DecoderBiAffine, self).__init__(config)
        self.config = config
        self.biaffine = BiAffine(config)

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.biaffine.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        inputs[self.config.output_map.logits] = self.biaffine(
            inputs[self.config.input_map.embedding]
        )
        return inputs
