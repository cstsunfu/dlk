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
from lightning.pytorch.callbacks import LearningRateMonitor
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, DictField, SubModules


@config_register("callback", 'lr_monitor')
@define
class LearningRateMonitorCallbackConfig(BaseConfig):
    name = NameField(value="lr_monitor", file=__file__, help="the learning rate monitor callback for lightning")
    @define
    class Config:
        logging_interval = StrField(value=None, checker=str_check(options=['step', 'epoch'], additions=None), help="""set to ``'epoch'`` or ``'step'`` to log ``lr`` of all optimizers at the same interval, set to ``None`` to log at individual interval according to the ``interval`` key of each scheduler. Defaults to ``None``.""")
        log_momentum = BoolField(value=False, help="option to also log the momentum values of the optimizer, if the optimizer has the ``momentum`` or ``betas`` attribute. Defaults to ``False``.")
    config = NestField(value=Config, converter=nest_converter)


@register("callback", 'lr_monitor')
class LearningRateMonitorCallback(object):
    """Monitor the learning rate
    """

    def __init__(self, config: LearningRateMonitorCallbackConfig):
        super().__init__()
        self.config = config.to_dict()['config']

    def __call__(self, rt_config: Dict)->LearningRateMonitor:
        """return LearningRateMonitor object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            LearningRateMonitor object

        """
        return LearningRateMonitor(**self.config)
