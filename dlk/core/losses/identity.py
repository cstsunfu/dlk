from typing import Dict
import torch.nn as nn
from . import loss_register, loss_config_register
from dlk.core.base_module import BaseModuleConfig
import torch.nn as nn


@loss_config_register("identity")
class IdentityLossConfig(BaseModuleConfig):
    """docstring for IdentityLossConfig
    {
        config: {
            "schedule": [1],
            "scale": [1], # scale the loss for every schedule
            // "schedule": [0.3, 1.0], # can be a list or str
            // "scale": "[0.5, 1]",
            "loss": "loss", // the real loss from result['loss']
        },
        _name: "identity",
    }
    """
    def __init__(self, config: Dict):
        super(IdentityLossConfig, self).__init__(config)
        config = config['config']

        self.scale = config['scale']
        self.schedule = config['schedule']
        self.loss = config['loss']

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
        self.post_check(config, used=[
            "loss",
            "schedule",
            "scale",
        ])

@loss_register("identity")
class IdentityLoss(object):
    def __init__(self, config: IdentityLossConfig):
        super(IdentityLoss, self).__init__()
        self.config = config

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
        loss = result[self.config.loss] * scale
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
