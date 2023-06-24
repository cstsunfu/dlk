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
from lightning.pytorch.callbacks import ModelCheckpoint
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, DictField, SubModules


@config_register("callback", 'checkpoint')
@define
class CheckpointCallbackConfig(BaseConfig):
    name = NameField(value="checkpoint", file=__file__, help="the checkpoint callback for lightning")
    @define
    class Config:
        monitor = StrField(value="*@*", help="the monitor metrics for checkpoint")
        dirpath = StrField(value=None, checker=str_check(additions=None), help="""
                           directory to save the model file.
                           By default, dirpath is ``None`` and will be set at runtime to the location
                           specified by :class:`~lightning.pytorch.trainer.trainer.Trainer`'s
                           :paramref:`~lightning.pytorch.trainer.trainer.Trainer.default_root_dir` argument,
                           and if the Trainer uses a logger, the path will also contain logger name and version.
                           """)
        filename = StrField(value=None, checker=str_check(additions=None), help="""
                            checkpoint filename. Can contain named formatting options to be auto-filled.
                            By default, filename is ``None`` and will be set to ``'{epoch}-{step}'``, where "epoch" and "step" match
                            the number of finished epoch and optimizer steps respectively.
                            """)
        save_top_k = IntField(value=3, checker=int_check(lower=0), help="""
                              if ``save_top_k == k``,
                              the best k models according to the quantity monitored will be saved.
                              if ``save_top_k == 0``, no models are saved.
                              if ``save_top_k == -1``, all models are saved.
                              Please note that the monitors are checked every ``every_n_epochs`` epochs.
                              if ``save_top_k >= 2`` and the callback is called multiple
                              times inside an epoch, the name of the saved file will be
                              appended with a version count starting with ``v1``.
                              """
                              )
        mode = StrField(value='max', checker=str_check(options=['min', 'max']), help="max or min for monitor metrics")
        save_last = BoolField(value=True, help="awlways save last checkpoint")
        auto_insert_metric_name = BoolField(value=True, help="the save file name with or not metric name")
        every_n_train_steps = IntField(value=None, checker=int_check(lower=0, additions=None), help="Number of training steps between checkpoints.")
        train_time_interval = IntField(value=None, checker=int_check(lower=0, additions=None), help="""
                                       Checkpoints are monitored at the specified time interval.
                                       For all practical purposes, this cannot be smaller than the amount
                                       of time it takes to process a single training batch. This is not
                                       guaranteed to execute at the exact time specified, but should be close.
                                       """)
        every_n_epochs = IntField(value=1, checker=int_check(lower=0, additions=None), help="Number of training epochs between checkpoints.")
        save_on_train_epoch_end = BoolField(value=False, help="Whether to run checkpointing at the end of the training epoch. If this is False, then the check runs at the end of the validation.")
        save_weights_only = BoolField(value=True, help="if ``True``, then only the model's weights will be saved. Otherwise, the optimizer states, lr-scheduler states, etc are added in the checkpoint too.")
        verbose = BoolField(value=True, help="verbosity mode.")
    config = NestField(value=Config, converter=nest_converter)

@register("callback", 'checkpoint')
class CheckpointCallback(object):
    """Save checkpoint decided by config
    """

    def __init__(self, config: CheckpointCallbackConfig):
        super().__init__()
        self.config = config.to_dict()['config']

    def __call__(self, rt_config: Dict)->ModelCheckpoint:
        """get the ModelCheckpoint object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            ModelCheckpoint object

        """
        dirpath = os.path.join(rt_config.get('save_dir', ''), rt_config.get("name", ''))
        return ModelCheckpoint(dirpath=dirpath, **self.config)
