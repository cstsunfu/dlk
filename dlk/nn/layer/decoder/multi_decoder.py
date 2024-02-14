# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Set

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
from dlk.utils.register import register, register_module_name


@cregister("decoder", "multi_decoder")
class MultiDecoderConfig(Base):
    """for multiple decoder"""

    submodule = SubModule({}, help="all the decode module config")


@register("decoder", "multi_decoder")
class MultiDecoder(SimpleModule):
    """multi_decoder a x A x b"""

    def __init__(self, config: MultiDecoderConfig):
        super(MultiDecoder, self).__init__(config)
        decode_configs = config._get_named_modules("decoder")
        self.decoders = nn.ModuleDict(
            {
                decode_name: register.get(
                    "decoder", register_module_name(decode_config._module_name)
                )(decode_config)
                for decode_name, decode_config in decode_configs.items()
            }
        )

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        for decode_name in self.decoders:
            self.decoders[decode_name].init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        for decode_name in self.decoders:
            inputs = self.decoders[decode_name](inputs)
        return inputs
