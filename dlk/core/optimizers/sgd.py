from typing import Dict
import torch.nn as nn
import torch.optim as optim
from dlk.utils.config import BaseConfig
from . import optimizer_register, optimizer_config_register, BaseOptimizer


@optimizer_config_register("sgd")
class SGDOptimizerConfig(BaseConfig):
    """
    {
        "config": {
            "lr": 1e-3,
            "momentum": 0.9,
            "dampening": 0,
            "weight_decay": 0,
            "nesterov":false,
            "optimizer_special_groups":[
            // special paramater groups set to special value, if some config key-value is not set, will use the default config in  optimizer_config.
            // You should sort the config by priority(
            //     e.g. the first group is ['linear.bias', {weight_decay: 0.1}], the second is [bias, [{weight_decay: 0.2}]], then the weight_decay of "*linea.bias*" will be 0.1, and the weight_decay of others *.bias.* will be 0.2
            // ["bias & LayerNorm.bias & LayerNorm.weight", {weight_decay: 0}]
            ]
        },
        "_name": "sgd",
    }
    """
    def __init__(self, config: Dict):
        super(SGDOptimizerConfig, self).__init__(config)
        self.config = config['config']
        self.post_check(self.config, used=[
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "optimizer_special_groups",
        ])


@optimizer_register("sgd")
class SGDOptimizer(BaseOptimizer):
    def __init__(self, model: nn.Module, config: SGDOptimizerConfig):
        super(SGDOptimizer, self).__init__()
        self.config = config.config
        self.model = model
        self.optimizer = optim.SGD

    def get_optimizer(self):
        """TODO: Docstring for get_optimizer.

        :arg1: TODO
        :returns: TODO
        """
        return self.init_optimizer(optim.SGD, self.model, self.config)
