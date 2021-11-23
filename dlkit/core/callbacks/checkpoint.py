import torch.nn as nn
from . import callback_register, callback_config_register
from typing import Dict, List
import os
from pytorch_lightning.callbacks import ModelCheckpoint
        

@callback_config_register('checkpoint')
class CheckpointCallbackConfig(object):
    """docstring for CheckpointCallbackConfig
    {
        // default checkpoint configure
        "_name": checkpoint,
        "config": {
            "monitor": null,
            "save_top_k": 3,
            "save_last": null,
            "auto_insert_metric_name": true,
            "every_n_train_steps": null,
            "every_n_epochs": null,
            "save_on_train_epoch_end": null,
            "save_weights_only": false,
        }
    }
    """
    def __init__(self, config):
        super(CheckpointCallbackConfig, self).__init__()
        config = config.get('config')
        self.monitor = config.get('monitor', None)
        self.save_last = config.get('save_last', None)
        self.save_top_k = config.get('save_top_k', 3)
        self.auto_insert_metric_name = config.get('auto_insert_metric_name', True)
        self.every_n_train_steps = config.get('every_n_train_steps', None)
        self.every_n_epochs = config.get('every_n_epochs', None)
        self.save_on_train_epoch_end = config.get('save_on_train_epoch_end', None)
        self.save_weights_only = config.get('save_weights_only', False)

@callback_register('checkpoint')
class CheckpointCallback(object):
    """
    """

    def __init__(self, config: CheckpointCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict):
        """TODO: Docstring for __call__.

        :rt_config: Dict: TODO
        :returns: TODO

        """
        dirpath = os.path.join(rt_config.get('save_dir', ''), rt_config.get("name", ''))
        return ModelCheckpoint(dirpath=dirpath, **self.config.__dict__)
