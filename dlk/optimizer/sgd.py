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


@cregister("optimizer", "sgd")
class SGDOptimizerConfig(BaseOptimizerConfig):
    """the sgd optimizer"""

    lr = FloatField(value=1e-3, minimum=0.0, help="the learning rate of optimizer")
    momentum = FloatField(value=0.9, minimum=0.0, help="the momentum of sgd")
    dampening = FloatField(value=0, minimum=0, help="the dampening of sgd")
    nesterov = BoolField(value=False, help="use nesterov of sgd or not")
    weight_decay = FloatField(
        value=0.0, minimum=0.0, help="the weight decay of the optimizer"
    )


@register("optimizer", "sgd")
class SGDOptimizer(BaseOptimizer):
    """wrap for optim.SGD"""

    def __init__(self, model: nn.Module, config: SGDOptimizerConfig):
        super(SGDOptimizer, self).__init__(model, config, optim.SGD)
