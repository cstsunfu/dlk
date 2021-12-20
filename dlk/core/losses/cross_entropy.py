from typing import Dict
import torch.nn as nn
from . import loss_register, loss_config_register
from packaging import version
import torch.nn as nn
from dlk.utils.logger import Logger
from dlk.core.base_module import BaseModuleConfig
import torch

logger = Logger.get_logger()

@loss_config_register("cross_entropy")
class CrossEntropyLossConfig(BaseModuleConfig):
    """docstring for CrossEntropyLossConfig
    {
        "config": {
            "ignore_index": -1,
            "weight": null, # or a list of value for every class
            "label_smoothing": 0.0, # torch>=1.10
            "pred_truth_pair": [], # len(.) == 2, the 1st is the pred_name, 2nd is truth_name in __call__ inputs
            "schedule": [1],
            "scale": [1], # scale the loss for every schedule stage
            // "schdeule": [0.3, 1.0], # can be a list or str
            // "scale": "[0.5, 1]",
        },
        "_name": "cross_entropy",
    }
    """
    def __init__(self, config: Dict):
        super(CrossEntropyLossConfig, self).__init__(config)
        config = config['config']

        self.scale = config['scale']
        self.schedule = config['schedule']

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

        self.weight = config['weight']
        self.ignore_index = config['ignore_index']
        self.label_smoothing = config['label_smoothing']
        self.pred_truth_pair = config['pred_truth_pair']
        if not self.pred_truth_pair:
            raise PermissionError(f"You must provide the pred_truth_pair for loss.")
        self.post_check(config, used=[
            "ignore_index",
            "weight",
            "label_smoothing",
            "pred_truth_pair",
            "schedule",
            "scale",
        ])


@loss_register("cross_entropy")
class CrossEntropyLoss(object):
    def __init__(self, config: CrossEntropyLossConfig):
        super(CrossEntropyLoss, self).__init__()
        self.config = config
        if (version.parse(torch.__version__)>=version.parse("1.10")):
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=config.weight,
                ignore_index=config.ignore_index,
                label_smoothing=config.label_smoothing
            )
        else:
            if config.label_smoothing:
                logger.info("Torch version is <1.10, so ignore label_smoothing")
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=config.weight,
                ignore_index=config.ignore_index,
            )

    def update_config(self, rt_config):
        """TODO: Docstring for update_config.
        :rt_config: TODO
         {
             "total_steps": self.num_training_steps,
             "total_epochs": self.num_training_epochs
         }
        :returns: TODO

        """
        self.current_stage = 0
        self.config.schedule = [rt_config['total_steps']*i for i in self.config.schedule]


    def calc(self, result, inputs, rt_config):
        """TODO: Docstring for get_loss.
        :returns: TODO
        """
        if rt_config['current_step']>self.config.schedule[self.current_stage]:
            self.current_stage += 1
        scale = self.config.scale[self.current_stage]
        pred_name, truth_name = self.config.pred_truth_pair
        pred = result[pred_name]
        target = inputs[truth_name]
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1)
        loss = self.cross_entropy(pred, target) * scale
        return loss

    def __call__(self, result, inputs, rt_config):
        """TODO: Docstring for __call__.
        :returns: TODO
         rt_config={
             "current_step": self.global_step,
             "current_epoch": self.current_epoch,
             "total_steps": self.num_training_steps,
             "total_epochs": self.num_training_epochs
         }),
        """
        return self.calc(result, inputs, rt_config)
