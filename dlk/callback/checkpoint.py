# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List

import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)
from lightning.pytorch.callbacks import ModelCheckpoint

from dlk.utils.register import register


@cregister("callback", "checkpoint")
class CheckpointCallbackConfig(Base):
    """callback for save checkpoint on the condition"""

    monitor = StrField(value="???", help="the monitor metrics for checkpoint")
    filename = StrField(
        value=None,
        additions=[None],
        help="""
        checkpoint filename. Can contain named formatting options to be auto-filled.
        By default, filename is ``None`` and will be set to ``'{epoch}-{step}'``, where "epoch" and "step" match
        the number of finished epoch and optimizer steps respectively.
        """,
    )
    save_top_k = IntField(
        value=3,
        minimum=0,
        help="""
        if ``save_top_k == k``,
        the best k models according to the quantity monitored will be saved.
        if ``save_top_k == 0``, no models are saved.
        if ``save_top_k == -1``, all models are saved.
        Please note that the monitors are checked every ``every_n_epochs`` epochs.
        if ``save_top_k >= 2`` and the callback is called multiple
        times inside an epoch, the name of the saved file will be
        appended with a version count starting with ``v1``.
        """,
    )
    mode = StrField(
        value="max",
        options=["min", "max"],
        help="max or min for monitor metrics",
    )
    save_last = BoolField(value=True, help="awlways save last checkpoint")
    auto_insert_metric_name = BoolField(
        value=True, help="the save file name with or not metric name"
    )
    every_n_train_steps = IntField(
        value=None,
        minimum=0,
        additions=[None],
        help="Number of training steps between checkpoints, default use `every_n_epochs`, when set the `every_n_train_steps` will disable the `every_n_epochs`.",
    )
    train_time_interval = IntField(
        value=None,
        minimum=0,
        additions=[None],
        help="""
        Checkpoints are monitored at the specified time interval.
        For all practical purposes, this cannot be smaller than the amount
        of time it takes to process a single training batch. This is not
        guaranteed to execute at the exact time specified, but should be close.
        when set the `train_time_interval` will disable the `every_n_epochs` and `every_n_train_steps`.
        """,
    )
    every_n_epochs = IntField(
        value=1,
        minimum=0,
        help="Number of training epochs between checkpoints.",
    )
    save_on_train_epoch_end = BoolField(
        value=False,
        help="Whether to run checkpointing at the end of the training epoch. If this is False, then the check runs at the end of the validation.",
    )
    save_weights_only = BoolField(
        value=True,
        help="if ``True``, then only the model's weights will be saved. Otherwise, the optimizer states, lr-scheduler states, etc are added in the checkpoint too.",
    )
    verbose = BoolField(value=True, help="verbosity mode.")
    enable_version_counter = BoolField(value=False, help="enable version counter")


@register("callback", "checkpoint")
class CheckpointCallback(object):
    """callback for save checkpoint on the condition"""

    def __init__(self, config: CheckpointCallbackConfig):
        super().__init__()
        self.config = config._to_dict(only_para=True)
        if self.config["train_time_interval"] is not None:
            self.config["every_n_epochs"] = None
            self.config["every_n_train_steps"] = None

        if self.config["every_n_train_steps"] is not None:
            self.config["every_n_epochs"] = None

    def __call__(self, rt_config: Dict) -> ModelCheckpoint:
        """get the ModelCheckpoint object

        Args:
            rt_config: runtime config, include log_dir, and the checkpoint path name

        Returns:
            ModelCheckpoint object

        """
        dirpath = os.path.join(
            rt_config.get("log_dir", ""), rt_config.get("name", ""), "checkpoint"
        )
        return ModelCheckpoint(dirpath=dirpath, **self.config)
