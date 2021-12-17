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
    def __init__(self, config: Dict):
        super(CheckpointCallbackConfig, self).__init__()
        config = config['config']
        self.monitor = config['monitor']
        self.save_last = config['save_last']
        self.save_top_k = config['save_top_k']
        self.auto_insert_metric_name = config['auto_insert_metric_name']
        self.every_n_train_steps = config['every_n_train_steps']
        self.every_n_epochs = config['every_n_epochs']
        self.save_on_train_epoch_end = config['save_on_train_epoch_end']
        self.save_weights_only = config['save_weights_only']

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
