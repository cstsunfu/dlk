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
from lightning.pytorch.callbacks import EarlyStopping

from dlk.utils.register import register


@cregister("callback", "early_stop")
class EarlyStoppingCallbackConfig(Base):
    """callback for the early stopping"""

    monitor = StrField(value=MISSING, help="the monitor metrics for checkpoint")
    mode = StrField(
        value="max",
        options=["min", "max"],
        help="max or min for monitor metrics",
    )
    patience = IntField(value=3, minimum=0, help="patience times for early stop")
    min_delta = FloatField(
        value=0.0,
        help="trigger early stop when the monitor metrics change less than min_delta",
    )
    check_on_train_epoch_end = BoolField(
        value=False,
        help="whether to run early stopping at the end of the training epoch.If this is ``False``, then the check runs at the end of the validation.",
    )
    strict = BoolField(
        value=True,
        help="whether to crash the training if `monitor` is not found in the validation metrics.",
    )
    stopping_threshold = FloatField(
        value=None,
        additions=[None],
        help="Stop training immediately once the monitored quantity reaches this threshold.",
    )
    divergence_threshold = FloatField(
        value=None,
        additions=[None],
        help="Stop training as soon as the monitored quantity becomes worse than this threshold.",
    )
    verbose = BoolField(value=True, help="verbosity mode.")
    log_rank_zero_only = BoolField(
        value=True,
        help="When set ``True``, logs the status of the early stopping callback only for rank 0 process.",
    )


@register("callback", "early_stop")
class EarlyStoppingCallback(object):
    """Early stop decided by config"""

    def __init__(self, config: EarlyStoppingCallbackConfig):
        super().__init__()
        self.config = config._to_dict(only_para=True)

    def __call__(self, rt_config: Dict) -> EarlyStopping:
        """return EarlyStopping object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns:
            EarlyStopping object

        """
        return EarlyStopping(**self.config)
