# Copyright 2021 cstsunfu. All rights reserved.
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

# TODO: WIP  comming soon!!
from typing import Dict
import torch.nn as nn
from . import loss_register, loss_config_register
import torch.nn as nn
from dlk.utils.config import ConfigTool


@loss_config_register("multi_loss")
class MultiLossConfig(object):
    """Config for MultiLoss

    Config Example:
        >>> {
        >>>     "loss@the_first": {
        >>>         config: {
        >>>             "ignore_index": -1,
        >>>             "weight": null, # or a list of value for every class
        >>>             "label_smoothing": 0.0, # torch>=1.10
        >>>             "pred_truth_pair": ["logits1", "label1"], # len(.) == 2, the 1st is the pred_name, 2nd is truth_name in __call__ inputs
        >>>             "schedule": [0.3, 0.6, 1],
        >>>             "scale": [1, 0, 0.5], # scale the loss for every schedule
        >>>             // "schdeule": [0.3, 1.0],
        >>>             // "scale": [0, 1, 0.5], # scale the loss
        >>>         },
        >>>         _name: "cross_entropy",
        >>>     },
        >>>     "loss@the_second": {
        >>>         config: {
        >>>             "pred_truth_pair": ["logits2", "label2"], # len(.) == 2, the 1st is the pred_name, 2nd is truth_name in __call__ inputs
        >>>             "schdeule": [0.3, 0.6, 1],
        >>>             "scale": [0, 1, 0.5], # scale the loss for every schedule
        >>>             // "schdeule": [0.3, 1.0],
        >>>             // "scale": [0, 1, 0.5], # scale the loss
        >>>         },
        >>>         _base: "cross_entropy",  // _name or _base is all ok
        >>>     },
        >>>     config: {
        >>>         "loss_list": ['the_first', 'the_second'],
        >>>     },
        >>>     _name: "cross_entropy",
        >>> }
    """
    def __init__(self, config: Dict):
        super(MultiLossConfig, self).__init__()
        config = config.get('config', {})



@loss_register("multi_loss")
class MultiLoss(object):
    """ This module is NotImplemented yet don't use it
    """
    def __init__(self, config: MultiLossConfig):
        super(MultiLoss, self).__init__()
        raise NotImplementedError
        self.config = config

    def get_loss(self, config):
        """Use config to init the loss

        Args:
            config: loss config

        Returns: 
            the Loss and the LossConfig

        """
        return ConfigTool.get_leaf_module(loss_register, loss_config_register, "loss", config)

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
        loss = 0
        for i, (pred, truth) in enumerate(self.config.pred_truth_pair):
            loss = loss + self.cross_entropy(result[pred], inputs[truth]) * self.config.loss_scale[i]
        return loss

    def __call__(self, result, inputs, rt_config):
        """same as self.calc
        """
        return self.calc(result, inputs, rt_config)
