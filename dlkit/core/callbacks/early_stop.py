import torch.nn as nn
from . import callback_register, callback_config_register
from typing import Dict, List
import os
from pytorch_lightning.callbacks import EarlyStopping


@callback_config_register('early_stop')
class EarlyStoppingCallbackConfig(object):
    """docstring for EarlyStoppingCallbackConfig
        {
            "_name": "early_stop",
            "config":{
                "monitor": "val_loss",
                "patience": 3,
                "min_delta": 0.0,
                "check_on_train_epoch_end": null,
                "strict": true,
            }
        }
    """
    def __init__(self, config: Dict):
        super(EarlyStoppingCallbackConfig, self).__init__()
        config = config['config']
        self.monitor = config['monitor']
        self.min_delta = config['min_delta']
        self.patience = config["patience"]
        self.check_on_train_epoch_end = config["check_on_train_epoch_end"]
        self.strict = config["strict"]

@callback_register('early_stop')
class EarlyStoppingCallback(object):
    """
    """

    def __init__(self, config: EarlyStoppingCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict):
        """TODO: Docstring for __call__.

        :rt_config: Dict: TODO
        :returns: TODO

        """
        return EarlyStopping(**self.config.__dict__)
