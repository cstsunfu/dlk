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
from pytorch_lightning.callbacks import EarlyStopping


@callback_config_register('early_stop')
class EarlyStoppingCallbackConfig(object):
    """Config for EarlyStoppingCallback

    Config Example:
        >>> {
        >>>     "_name": "early_stop",
        >>>     "config":{
        >>>         "monitor": "val_loss",
        >>>         "mode": "*@*", // min or max, min for the monitor is loss, max for the monitor is acc, f1, etc.
        >>>         "patience": 3,
        >>>         "min_delta": 0.0,
        >>>         "check_on_train_epoch_end": null,
        >>>         "strict": true, // if the monitor is not right, raise error
        >>>         "stopping_threshold": null, // float, if the value is good enough, stop
        >>>         "divergence_threshold": null, // float,  if the value is so bad, stop
        >>>         "verbose": true, //verbose mode print more info
        >>>     }
        >>> }
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
    """Early stop decided by config
    """

    def __init__(self, config: EarlyStoppingCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict)->EarlyStopping:
        """return EarlyStopping object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            EarlyStopping object

        """
        return EarlyStopping(**self.config.__dict__)
