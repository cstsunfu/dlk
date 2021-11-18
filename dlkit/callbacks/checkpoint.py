import torch.nn as nn
from . import callback_register, callback_config_register
from typing import Dict, List
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint
        

@callback_config_register('checkpoint')
class CheckpointCallbackConfig(object):
    """docstring for CheckpointCallbackConfig
    """
    def __init__(self, config):
        super(CheckpointCallbackConfig, self).__init__()
        config = config.get('config')
        self.
        dirpath=None, filename=None, monitor=None, verbose=False, save_last=None, save_top_k=1, save_weights_only=False, mode='min', auto_insert_metric_name=True, every_n_train_steps=None, train_time_interval=None, every_n_epochs=None, save_on_train_epoch_end=None, every_n_val_epochs=Non


@callback_register('checkpoint')
class CheckpointCallback(object):
    """
    """

    def __init__(self, config: CheckpointCallbackConfig, rt_config: Dict):
        super().__init__()
        
