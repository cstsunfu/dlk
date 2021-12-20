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
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-6,
            "weight_decay": 1e-2,
            "optimizer_special_groups": {
                "order": ['decoder', 'bias'], // the group order, if the para is in decoder & is in bias, set to decoder
                "bias": {
                    "config": {
                        "weight_decay": 0
                    },
                    "pattern": ["bias",  "LayerNorm.bias", "LayerNorm.weight"]
                },
                "decoder": {
                    "config": {
                        "lr": 1e-3
                    },
                    "pattern": ["decoder"]
                },
            }
            "name": "default" // default group name
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
            "name",
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
