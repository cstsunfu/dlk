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
from dlk.nn.module.linear import Linear, LinearConfig
from dlk.utils.register import register


@cregister("encoder", "linear")
class EncoderLinearConfig(LinearConfig):
    """the linear encoder module"""

    class InputMap:
        embedding = StrField(value="embedding", help="the input name of embedding")

    input_map = NestField(value=InputMap, help="the input map of the linear module")

    class OutputMap:
        embedding = StrField(value="embedding", help="the output name of embedding")

    output_map = NestField(value=OutputMap, help="the output map of the linear module")


@register("encoder", "linear")
class EncoderLinear(SimpleModule):
    """wrap for torch.nn.Linear"""

    def __init__(self, config: EncoderLinearConfig):
        super(EncoderLinear, self).__init__(config)
        self.linear = register.get("module", "linear")(config)
        self.config = config

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.linear.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """All step do this

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        inputs[self.config.output_map.embedding] = self.linear(
            inputs[self.config.input_map.embedding]
        )
        return inputs
