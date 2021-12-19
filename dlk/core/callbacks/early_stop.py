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
                "mode": "*@*", // min or max, min for the monitor is loss, max for the monitor is acc, f1, etc.
                "patience": 3,
                "min_delta": 0.0,
                "check_on_train_epoch_end": null,
                "strict": true, // if the monitor is not right, raise error
                "stopping_threshold": null, // float, if the value is good enough, stop
                "divergence_threshold": null, // float,  if the value is so bad, stop
                "verbose": true, //verbose mode print more info
            }
        }
    """
    def __init__(self, config: Dict):
        super(EarlyStoppingCallbackConfig, self).__init__()
        config = config['config']
        self.monitor = config['monitor']
        self.mode = config['mode']
        self.patience = config["patience"]
        self.min_delta = config['min_delta']
        self.strict = config['strict']
        self.verbose = config['verbose']
        self.stopping_threshold = config['stopping_threshold']
        self.divergence_threshold = config['divergence_threshold']
        self.check_on_train_epoch_end = config["check_on_train_epoch_end"]

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
