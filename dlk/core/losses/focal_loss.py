# Copyright cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
from . import loss_register, loss_config_register
from packaging import version
from dlk.utils.logger import Logger
from dlk.core.base_module import BaseModuleConfig
import torch

logger = Logger.get_logger()

@loss_config_register("focal_loss")
class FocalLossLossConfig(BaseModuleConfig):
    default_config = {
        "_name": "focal_loss",
        "config": {
            "ignore_index": -100,
            "weight": None, # or a list of value for every class
            "pred_truth_pair": [], # len(.) == 2, the 1st is the pred_name, 2nd is truth_name in __call__ inputs . OR the elements are list too it means there are multiple pred_truth_pairs
            "log_map": {
                "loss": "loss"
            },
            "schedule": [1],
            "gamma": 1.0, # focal loss gamma
            "reduction": "mean",
            "scale": [1], # scale the loss for every schedule stage
            # "schdeule": [0.3, 1.0], # can be a list or str
            # "scale": "[0.5, 1]",
        },
    }
    """Config for FocalLossLoss

    Config Example: default_config
    """
    def __init__(self, config: Dict):
        super(FocalLossLossConfig, self).__init__(config)
        config = config['config']

        self.scale = config['scale']
        self.schedule = config['schedule']
        self.reduction = config['reduction']
        self.gamma = config['gamma']

        if isinstance(self.scale, str):
            self.scale = eval(self.scale)
        if isinstance(self.schedule, str):
            self.schedule = eval(self.schedule)

        if not isinstance(self.scale, list):
            assert isinstance(float(self.scale), float)
            self.scale = [self.scale]
        if not isinstance(self.schedule, list):
            assert isinstance(float(self.schedule), float)
            self.schedule = [self.schedule]
        assert len(self.schedule) == len(self.scale)
        assert self.schedule[-1] - 1 < 0.00001

        self.log_map = config['log_map']
        if isinstance(self.log_map, str):
            self.log_map = {"loss": self.log_map}
        self.weight = config['weight']
        self.ignore_index = config['ignore_index']
        self.label_smoothing = config['label_smoothing']
        self.pred_truth_pair = config['pred_truth_pair']
        if not self.pred_truth_pair:
            raise PermissionError(f"You must provide the pred_truth_pair for loss.")
        self.post_check(config, used=[
            "ignore_index",
            "weight",
            "pred_truth_pair",
            "schedule",
            "scale",
            "log_map"
        ])


@loss_register("focal_loss")
class FocalLossLoss(nn.Module):
    """for multi class classification
    """
    def __init__(self, config: FocalLossLossConfig):
        super(FocalLossLoss, self).__init__()
        self.config = config
        self.weight = None
        if config.weight:
            self.weight = torch.tensor(config.weight, dtype=torch.float)

    def update_config(self, rt_config):
        """callback for imodel to update the total steps and epochs

        when init the loss module, the total step and epoch is not known, when all data ready, the imodel update the value for loss module

        Args:
            rt_config: { "total_steps": self.num_training_steps, "total_epochs": self.num_training_epochs}

        Returns: 
            None

        """
        self.current_stage = 0
        self.config.schedule = [rt_config['total_steps']*i for i in self.config.schedule]

    def focal_loss(self, pred, target):
        """calc focal loss

        Args:
            pred: NxC for logits
            target: N for target

        Returns: 
            loss

        """
        # drop the ignore target
        mask = target != self.config.ignore_index
        target = target[mask]
        pred = pred[mask]
        # [N, 1]
        target = target.unsqueeze(-1)
        # [N, C]
        pt = F.softmax(pred, dim=-1)
        logpt = F.log_softmax(pred, dim=-1)
        # [N]
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        if self.weight is not None:
            # [N] at[i] = weight[target[i]]
            at = self.weight.gather(0, target.squeeze(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.config.gamma * logpt
        if self.config.reduction == "none":
            return loss
        if self.config.reduction == "mean":
            return loss.mean()
        return loss.sum()

    def calc(self, result, inputs, rt_config):
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

        Returns: 
            loss

        """
        if rt_config['current_step']>self.config.schedule[self.current_stage]:
            self.current_stage += 1
        scale = self.config.scale[self.current_stage]
        pred_name, truth_name = self.config.pred_truth_pair
        pred = result[pred_name]
        target = inputs[truth_name]
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1)

        loss = self.focal_loss(pred, target) * scale
        return loss, {self.config.log_map['loss']: loss}

    def __call__(self, result, inputs, rt_config):
        """same as self.calc
        """
        return self.calc(result, inputs, rt_config)
