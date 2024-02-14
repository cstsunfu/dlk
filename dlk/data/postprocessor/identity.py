# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union

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

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
from dlk.utils.register import register


@cregister("postprocessor", "identity")
class IdentityPostProcessorConfig(BasePostProcessorConfig):
    """identity postprocessor"""

    pass


@register("postprocessor", "identity")
class IdentityPostProcessor(BasePostProcessor):
    """identity postprocessor"""

    def __init__(self, config: IdentityPostProcessorConfig):
        super(IdentityPostProcessor, self).__init__(config)

    def process(self, stage, outputs, origin_data) -> Dict:
        """do nothing except gather the loss"""
        if "loss" in outputs:
            return {self.loss_name_map(stage): torch.mean(outputs["loss"])}
        return {}
