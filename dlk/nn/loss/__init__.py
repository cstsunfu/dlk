# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
from typing import Dict

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
    dataclass,
)

from dlk.utils.import_module import import_module_dir


@dataclass
class BaseLossConfig(Base):
    """the base loss config"""

    schedule = ListField(value=[1], help="the schedule of the loss, works with scale")
    scale = ListField(value=[1], help="the scale of the loss for every schedule stage")
    pred_truth_pair = DictField(
        value={},
        suggestions=[{"logits": "label_ids"}],
        help="""
        it's a dict of [predit_logits, truth_target] pair.
        If you have more than one pair,
        you should use the `multi_loss` module and provide this module as the submodule.
        """,
    )
    reduction = StrField(
        value="mean",
        options=["mean", "sum", "none"],
        help="the reduction method, support 'mean', 'sum', 'none'",
    )

    class LogMap:
        loss = StrField(value="loss", help="the output loss name")

    log_map = NestField(value=LogMap, help="the output loss name")


class BaseLoss(nn.Module):
    def __init__(self, config: BaseLossConfig):
        super(BaseLoss, self).__init__()
        self.config = config
        if len(self.config.pred_truth_pair) == 1:
            self.pred_name = list(self.config.pred_truth_pair.keys())[0]
            self.truth_name = self.config.pred_truth_pair[self.pred_name]
        else:
            assert len(self.config.pred_truth_pair) < 1

    def update_config(self, rt_config: Dict):
        """callback for imodel to update the total steps and epochs

        when init the loss module, the total step and epoch is not known, when all data ready, the imodel update the value for loss module

        Args:
            rt_config: { "total_steps": self.num_training_steps, "total_epochs": self.num_training_epochs}

        Returns:
            None

        """
        self.current_stage = 0
        self.config.schedule = [
            rt_config["total_steps"] * i for i in self.config.schedule
        ]

    def _calc(self, result, inputs, rt_config, scale):
        """calc the loss the predict is from result, the ground truth is from inputs

        Args:
            result: the model predict dict
            inputs: the all inputs for model
            rt_config: provide the current training status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }
            scale: the scale rate for the loss

        Returns:
            loss

        """
        raise NotImplementedError

    def forward(self, result, inputs, rt_config):
        """same as self.calc"""
        if rt_config["current_step"] > self.config.schedule[self.current_stage]:
            self.current_stage += 1
        scale = self.config.scale[self.current_stage]
        return self._calc(result, inputs, rt_config, scale)


loss_dir = os.path.dirname(__file__)
import_module_dir(loss_dir, "dlk.nn.loss")
