# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
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
from lightning import Trainer as PLTrainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.functional import Tensor

from dlk.utils.get_root import get_root
from dlk.utils.io import open
from dlk.utils.register import register, register_module_name

ogger = logging.getLogger(__name__)


@cregister("trainer", "lightning")
class LightningTrainerConfig(Base):
    """the basic trainer config. aka lightning trainer"""

    accelerator = StrField(
        value="auto",
        options=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"],
        help="""
        Supports passing different accelerator types
        ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto") as well as custom accelerator instances. 
        """,
    )
    strategy = StrField(
        value="auto",
        options=[
            "auto",
            "ddp",
            "ddp_notebook",
            "ddp_find_unused_parameters_true",
            "ddp_spawn",
            "deepspeed",
            "deepspeed_stage_1",
            "deepspeed_stage_2",
            "deepspeed_stage_2_offload",
            "deepspeed_stage_3",
            "deepspeed_stage_3_offload",
            "deepspeed_stage_3_offload_nvme",
            "fsdp",
            "fsdp_cpu_offload",
        ],
        help="""
        Supports passing different training strategies with aliases as well custom strategies. 
        you can create a strategy by yourself and register it 
        to the strategy registry`lightning.pytorch.strategies.StrategyRegistry`.
        """,
    )
    devices = AnyField(
        value="auto",
        help="""
        The devices to use.
        Can be set to a positive number (int or str),
        a sequence of device indices (list or str),
        the value ``-1`` to indicate all available devices should be used,
        or ``"auto"`` for automatic selection based on the chosen accelerator.
        Default: ``"auto"``.
        """,
    )
    precision = StrField(
        value="32-true",
        options=[
            "64",
            "64-true",
            "32",
            "32-true",
            "16",
            "16-mixed",
            "bf16",
            "bf16-mixed",
        ],
        help=""" 
        Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: ``'32-true'``. 
        """,
    )
    logger = BoolField(
        value=True,
        help=""" Enable logger or not. Default: ``True``. """,
    )
    fast_dev_run = BoolField(
        value=False,
        help=""" Runs 1 batch of train, val and test to find any bugs (ie: a sort of unit test). Default: ``False``. """,
    )
    max_epochs = IntField(
        value=1000,
        help="""
        Stop training once this number of epochs is reached.
        Disabled by default (None). 
        If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000. 
        To enable infinite training, set ``max_epochs`` = -1. 
        """,
    )
    min_epochs = IntField(
        value=None,
        additions=[None],
        help=""" Force training for at least these many epochs. Disabled by default (None). """,
    )
    max_steps = IntField(
        value=-1,
        help=""" 
        Stop training after this number of steps.
        Disabled by default (None). 
        If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.
        To enable infinite training, set ``max_epochs`` = -1. """,
    )
    min_steps = IntField(
        value=None,
        additions=[None],
        help=""" Force training for at least these number of steps. Disabled by default (None). """,
    )
    max_time = StrField(
        value=None,
        additions=[None],
        help=""" 
        Stop training after this amount of time has passed.
        Disabled by default (None).
        The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds),
        as a :class:`datetime.timedelta`, or a :class:`numpy.timedelta64`. 
        """,
    )
    limit_train_batches = FloatField(
        value=1.0,
        minimum=0.0,
        help="""How much of training dataset to check (percent). Default: ``1.0``.""",
    )
    limit_val_batches = FloatField(
        value=1.0,
        minimum=0.0,
        help=""" How much of validation dataset to check (percent). Default: ``1.0``.""",
    )
    limit_test_batches = FloatField(
        value=1.0,
        minimum=0.0,
        help=""" How much of test dataset to check (percent). Default: ``1.0``.
                                     """,
    )
    limit_predict_batches = FloatField(
        value=1.0,
        minimum=0.0,
        help=""" How much of prediction dataset to check (percent). Default: ``1.0``. """,
    )
    overfit_batches = FloatField(
        value=0.0,
        minimum=0.0,
        help="""Overfit a fraction of training/validation data (float) or a set number of batches (int). """,
    )
    val_check_interval = FloatField(
        value=1.0,
        help=""" How often to check the validation set. Use float to check within a training epoch. Default: ``1.0``. """,
    )
    check_val_every_n_epoch = IntField(
        value=1,
        additions=[None],
        help=""" 
        Perform a validation loop every `N` training epochs. 
        If ``None``, validation will be done solely based on the number of training batches, requiring ``val_check_interval`` to be an integer value. Default: ``1``. """,
    )
    num_sanity_val_steps = IntField(
        value=2,
        minimum=-1,
        help=""" 
        Sanity check runs n validation batches before starting the training routine.
        Set it to `-1` to run all batches in all validation dataloaders.
        Default: ``2``. 
        """,
    )
    log_every_n_steps = IntField(
        value=50,
        minimum=1,
        help="""How often to log within steps. Default: ``50``. """,
    )
    enable_progress_bar = BoolField(
        value=True,
        help="""Whether to enable to progress bar by default. Default: ``True``. """,
    )
    enable_model_summary = BoolField(
        value=True,
        help=""" Whether to enable model summarization by default. Default: ``True``. """,
    )
    accumulate_grad_batches = IntField(
        value=1,
        help="""Accumulates gradients over k batches before stepping the optimizer. Default: 1.""",
    )
    gradient_clip_val = FloatField(
        value=5.0,
        additions=[None],
        help=""" 
        The value at which to clip gradients.
        Passing ``gradient_clip_val=None`` disables gradient clipping.
        If using Automatic Mixed Precision (AMP), the gradients will be unscaled before..
        """,
    )
    gradient_clip_algorithm = StrField(
        value="norm",
        options=["norm", "value"],
        help=""" 
        The gradient clipping algorithm to use.
        Pass ``gradient_clip_algorithm="value"`` to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm.
        By default it will be set to ``"norm"``.
        """,
    )
    deterministic = BoolField(
        value=False,
        help=""" 
        If ``True``, sets whether PyTorch operations must use deterministic algorithms.
        Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations that don't support deterministic mode (requires PyTorch 1.11+).
        """,
    )
    inference_mode = BoolField(
        value=True,
        help=""" Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during evaluation (``validate``/``test``/``predict``).""",
    )
    use_distributed_sampler = BoolField(
        value=True,
        help=""" 
        Whether to wrap the DataLoader's sampler with :class:`torch.utils.data.DistributedSampler`.
        If not specified this is toggled automatically for strategies that require it.
        By default, it will add ``shuffle=True`` for the train sampler and ``shuffle=False`` for validation/test/predict samplers.
        If you want to disable this logic, you can pass ``False`` and add your own distributed sampler in the dataloader hooks.
        If ``True`` and a distributed sampler was already added, Lightning will not replace the existing one.
        For iterable-style datasets, we don't do this automatically. 
        """,
    )
    plugins = ListField(
        value=None,
        additions=[None],
        help="""
        Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
        Default: ``None``.
        """,
    )
    sync_batchnorm = BoolField(
        value=False,
        help="""Synchronize batch norm layers between process groups/whole world. Default: ``False``.""",
    )
    reload_dataloaders_every_n_epochs = IntField(
        value=0,
        minimum=0,
        help=""" Set to a non-negative integer to reload dataloaders every n epochs. Default: ``0``. """,
    )

    submodule = SubModule(
        {},
        help=""" the callback modules configure""",
    )


@register("trainer", "lightning")
class LightningTrainer(object):
    """pytorch-lightning trainer"""

    def __init__(self, config: LightningTrainerConfig, rt_config: Dict):
        super().__init__()
        config_dict = config._to_dict(only_para=True)
        if config_dict["logger"]:
            config_dict["logger"] = TensorBoardLogger(
                save_dir=os.path.join(rt_config["log_dir"], rt_config["name"]),
                name="metrics",
                version="",
                default_hp_metric=False,
            )
        callbacks = []
        callback_configs = config._get_named_modules("callback")
        config_dict["enable_checkpointing"] = False
        for callback_config_name in callback_configs:
            config_dict.pop(f"@{callback_config_name}")
            callback_config = callback_configs[callback_config_name]
            if callback_config._module_name == "checkpoint":
                config_dict["enable_checkpointing"] = True
            callbacks.append(
                register.get(
                    "callback", register_module_name(callback_config._module_name)
                )(callback_config)(rt_config=rt_config)
            )

        config_dict["callbacks"] = callbacks
        config_dict["default_root_dir"] = rt_config["log_dir"]
        self.trainer = PLTrainer(**config_dict)

    @property
    def world_size(self):
        return self.trainer.world_size

    def fit(self, **inputs):
        """fit the model and datamodule to trainer

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns:
            Undefine

        """

        return self.trainer.fit(**inputs)

    def predict(self, **inputs):
        """fit the model and datamodule.predict_dataloader to predict

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns:
            predict list

        """
        return self.trainer.predict(**inputs)

    def test(self, **inputs):
        """fit the model and datamodule.test_dataloader to test

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns:
            Undefine

        """
        return self.trainer.test(**inputs)

    def validate(self, **inputs):
        """fit the model and datamodule.validation to validate

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns:
            Undefine

        """
        return self.trainer.validate(**inputs)
