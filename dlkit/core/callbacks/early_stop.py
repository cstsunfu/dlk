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
    def __init__(self, config):
        super(EarlyStoppingCallbackConfig, self).__init__()
        config = config.get('config')
        self.monitor = config.get('monitor', "val_loss")
        self.min_delta = config.get('min_delta', 0.0)
        self.patience = config.get("patience", 3)
        self.check_on_train_epoch_end = config.get("check_on_train_epoch_end", None)
        self.strict = config.get("strict", True)

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
