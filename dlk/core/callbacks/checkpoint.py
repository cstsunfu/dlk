# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from . import callback_register, callback_config_register
from typing import Dict, List
import os
from pytorch_lightning.callbacks import ModelCheckpoint


@callback_config_register('checkpoint')
class CheckpointCallbackConfig(object):
    """Config for CheckpointCallback

    Config Example:
        >>> {
        >>>     // default checkpoint configure
        >>>     "_name": "checkpoint",
        >>>     "config": {
        >>>         "monitor": "*@*",    // monitor which metrics or log value
        >>>         "save_top_k": 3,   //save top k
        >>>         "mode": "*@*", //"max" or "min" select topk min or max checkpoint, min for loss, max for acc
        >>>         "save_last": true,  //  always save last checkpoint
        >>>         "auto_insert_metric_name": true, //the save file name with or not metric name
        >>>         "every_n_train_steps": null, // Number of training steps between checkpoints.
        >>>         "every_n_epochs": 1, //Number of epochs between checkpoints.
        >>>         "save_on_train_epoch_end": false,// Whether to run checkpointing at the end of the training epoch. If this is False, then the check runs at the end of the validation.
        >>>         "save_weights_only": false, //whether save other status like optimizer, etc.
        >>>     }
        >>> }
    """
    def __init__(self, config: Dict):
        super(CheckpointCallbackConfig, self).__init__()
        config = config['config']
        self.monitor = config['monitor']
        self.save_last = config['save_last']
        self.save_top_k = config['save_top_k']
        self.mode = config['mode']
        self.auto_insert_metric_name = config['auto_insert_metric_name']
        self.every_n_train_steps = config['every_n_train_steps']
        self.every_n_epochs = config['every_n_epochs']
        self.save_on_train_epoch_end = config['save_on_train_epoch_end']
        self.save_weights_only = config['save_weights_only']

@callback_register('checkpoint')
class CheckpointCallback(object):
    """Save checkpoint decided by config
    """

    def __init__(self, config: CheckpointCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict)->ModelCheckpoint:
        """get the ModelCheckpoint object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            ModelCheckpoint object

        """
        dirpath = os.path.join(rt_config.get('save_dir', ''), rt_config.get("name", ''))
        return ModelCheckpoint(dirpath=dirpath, **self.config.__dict__)
