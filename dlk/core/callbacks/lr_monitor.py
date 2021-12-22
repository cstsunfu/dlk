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

from typing import Dict
from . import callback_register, callback_config_register
from pytorch_lightning.callbacks import LearningRateMonitor


@callback_config_register('lr_monitor')
class LearningRateMonitorCallbackConfig(object):
    """Config for LearningRateMonitorCallback

    Config Example:
        >>> {
        >>>     "_name": "lr_monitor",
        >>>     "config": {
        >>>         "logging_interval": null, // set to None to log at individual interval according to the interval key of each scheduler. other value : step, epoch
        >>>         "log_momentum": true, // log momentum or not
        >>>     }
        >>> }
    """
    def __init__(self, config: Dict):
        config = config['config']
        self.logging_interval = config["logging_interval"]
        self.log_momentum = config["log_momentum"]


@callback_register('lr_monitor')
class LearningRateMonitorCallback(object):
    """Monitor the learning rate
    """

    def __init__(self, config: LearningRateMonitorCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict)->LearningRateMonitor:
        """return LearningRateMonitor object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            LearningRateMonitor object

        """
        return LearningRateMonitor(**self.config.__dict__)
