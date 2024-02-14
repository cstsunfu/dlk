# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.nn as nn
import torch.optim as optim
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

from . import BaseOptimizer, BaseOptimizerConfig


@cregister("optimizer", "adamw")
class AdamWOptimizerConfig(BaseOptimizerConfig):
    """the AdamW optimizer"""

    lr = FloatField(value=1e-3, minimum=0.0, help="the learning rate of optimizer")
    betas = ListField(value=[0.9, 0.999], help="the betas of adamw")
    eps = FloatField(value=1e-8, minimum=1e-20, help="the epsilon of optimizer")
    weight_decay = FloatField(
        value=1e-2, minimum=0.0, help="the weight decay of the optimizer"
    )


@register("optimizer", "adamw")
class AdamWOptimizer(BaseOptimizer):
    """Wrap for optim.AdamW"""

    def __init__(self, model: nn.Module, config: AdamWOptimizerConfig):
        super(AdamWOptimizer, self).__init__(model, config, optim.AdamW)
