from typing import Dict
import torch.nn as nn
import torch.optim as optim
from dlk.utils.config import BaseConfig
from . import optimizer_register, optimizer_config_register, BaseOptimizer


@optimizer_config_register("adamw")
class AdamWOptimizerConfig(BaseConfig):
    """docstring for LinearConfig
    {
        "config": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-6,
            "weight_decay": 1e-2,
            "optimizer_special_groups":[
            // special paramater groups set to special value, if some config key-value is not set, will use the default config in  optimizer_config.
            // You should sort the config by priority(
            //     e.g. the first group is ['linear.bias', {weight_decay: 0.1}], the second is [bias, [{weight_decay: 0.2}]], then the weight_decay of "*linea.bias*" will be 0.1, and the weight_decay of others *.bias.* will be 0.2
                ["bias & LayerNorm.bias & LayerNorm.weight", {weight_decay: 0}]
            ]
        },
        "_name": "adamw",
    }
    """
    def __init__(self, config: Dict):
        super(AdamWOptimizerConfig, self).__init__(config)
        self.config = config['config']
        self.post_check(self.config, used=[
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "optimizer_special_groups",
        ])


@optimizer_register("adamw")
class AdamWOptimizer(BaseOptimizer):
    def __init__(self, model: nn.Module, config: AdamWOptimizerConfig):
        super(AdamWOptimizer, self).__init__()
        self.config = config.config
        self.model = model
        self.optimizer = optim.AdamW

    def get_optimizer(self):
        """TODO: Docstring for __call__.
        :returns: TODO

        """

        return self.init_optimizer(optim.AdamW, self.model, self.config)
