# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

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

from dlk.utils.register import register

from . import Module


@cregister("module", "logits_gather")
class LogitsGatherConfig:
    """the logits_gather config"""

    prefix = StrField(value="gather_logits_", help="the prefix of the output name")
    logits_name_map = StrField(
        value="logits", help="the logits name map, gather which column"
    )
    gather_layer = DictField(
        value={},
        help="""
        the gather layer config, like
        "gather_layer": {
            "0": {
                "map": "3", # the 0th layer not do scale output to "gather_logits_3", "gather_logits_" is the output name prefix, the "3" is map name
                },
            "1": {
                "map": "4",
                }
            },
        default is empty dict
        """,
    )


@register("module", "logits_gather")
class LogitsGather(Module):
    """Gather the output logits decided by config"""

    def __init__(self, config: LogitsGatherConfig):
        super(LogitsGather, self).__init__()
        self.config = config
        self.layer_map: Dict[str, str] = {}
        self.prefix = self.config.prefix
        for layer, layer_config in self.config.gather_layer.items():
            self.layer_map[str(layer)] = str(layer_config["map"])

        if not self.layer_map:
            self.pass_gather = True
        else:
            self.pass_gather = False

    def forward(self, input: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """gather the needed input to dict

        Args:
            batch: a mini batch inputs

        Returns:
            some elements to dict

        """
        result = torch.jit.annotate(Dict[str, torch.Tensor], {})

        if self.pass_gather:
            return result
        for layer, layer_suffix in self.layer_map.items():
            result[self.prefix + layer_suffix] = input[int(layer)]

        return result
