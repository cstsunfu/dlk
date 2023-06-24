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
from typing import Dict, List
import os
from lightning.pytorch.callbacks import EarlyStopping
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, DictField, SubModules


@config_register("callback", 'early_stop')
@define
class EarlyStoppingCallbackConfig(BaseConfig):
    name = NameField(value="early_stop", file=__file__, help="the early stop callback for lightning")
    @define
    class Config:
        monitor = StrField(value="*@*", help="the monitor metrics for checkpoint")
        mode = StrField(value='max', checker=str_check(options=['min', 'max']), help="max or min for monitor metrics")
        patience = IntField(value=3, checker=int_check(lower=0), help="patience times for early stop")
        min_delta = FloatField(value=0.0, checker=float_check(), help="trigger early stop when the monitor metrics change less than min_delta")
        check_on_train_epoch_end = BoolField(value=False, help="whether to run early stopping at the end of the training epoch.If this is ``False``, then the check runs at the end of the validation.")
        strict = BoolField(value=True, help="whether to crash the training if `monitor` is not found in the validation metrics.")
        stopping_threshold = FloatField(value=None, checker=float_check(additions=None), help="Stop training immediately once the monitored quantity reaches this threshold.")
        divergence_threshold = FloatField(value=None, checker=float_check(additions=None), help="Stop training as soon as the monitored quantity becomes worse than this threshold.")
        verbose = BoolField(value=True, help="verbosity mode.")
        log_rank_zero_only = BoolField(value=True, help="When set ``True``, logs the status of the early stopping callback only for rank 0 process.")
    config = NestField(value=Config, converter=nest_converter)


@register("callback", 'early_stop')
class EarlyStoppingCallback(object):
    """Early stop decided by config
    """

    def __init__(self, config: EarlyStoppingCallbackConfig):
        super().__init__()
        self.config = config.to_dict()['config']

    def __call__(self, rt_config: Dict)->EarlyStopping:
        """return EarlyStopping object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            EarlyStopping object

        """
        return EarlyStopping(**self.config)
