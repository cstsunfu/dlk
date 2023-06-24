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

from torch.functional import Tensor
import torch.nn as nn
from typing import Dict, List
import hjson
from dlk.utils.config import BaseConfig, ConfigTool
from dlk.utils.get_root import get_root
import os
from dlk.utils.io import open
import dlk.utils.parser as parser
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("trainer", 'basic')
@define
class LightningTrainerConfig(BaseConfig):
    name = NameField(value="basic", file=__file__, help="the basic trainer config. aka lightning trainer")
    @define
    class Config:
        accelerator = StrField(value="auto", checker=str_check(options=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"]), help="""
                               Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto") as well as custom accelerator instances.
                               """)
        strategy = StrField(value='auto', checker=str_check(options=[
            "ddp", "ddp_notebook", "ddp_fork", "ddp_spawn", "dp",
            "deepspeed", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "deepspeed_stage_3_offload", "deepspeed_stage_3_offload_nvme",
            "fsdp", "fsdp_cpu_offload"
            ]), help="""
                            Supports passing different training strategies with aliases as well custom strategies. 
                            """)
        devices = IntField(value="auto", checker=int_check(additions='auto'), help="""
                           The devices to use. Can be set to a positive number (int or str), a sequence of device indices (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for automatic selection based on the chosen accelerator. Default: ``"auto"``.
                           """)
        precision = StrField(value="32-true", checker=str_check(options=["64", "64-true", "32", "32-true", "16", "16-mixed", "bf16", "bf16-mixed"]), help="""
                             Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed'). Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: ``'32-true'``.
                             """)
        logger = BoolField(value=True, help="""
                           Enable logger or not. Default: ``True``.
                           """)
        callbacks = ListField(value=[], help="""
                              A list of callbacks to connect to the Trainer. Default: ``[]``.
                              """)
        fast_dev_run = BoolField(value=False, help="""
                                 Runs 1 batch of train, val and test to find any bugs (ie: a sort of unit test). Default: ``False``.
                                 """)
        max_epochs = IntField(value=1000, help="""
                              Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000. To enable infinite training, set ``max_epochs`` = -1.
                              """)
        min_epochs = IntField(value=None, checker=int_check(additions=None), help="""
                              Force training for at least these many epochs. Disabled by default (None).
                              """)
        max_steps = IntField(value=-1, help="""
                             Stop training after this number of steps. Disabled by default (None). If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000. To enable infinite training, set ``max_epochs`` = -1.
                             """)
        min_steps = IntField(value=None, checker=int_check(additions=None), help="""
                             Force training for at least these number of steps. Disabled by default (None).
                             """)
        max_time = StrField(value=None, checker=str_check(additions=None), help="""
                            Stop training after this amount of time has passed. Disabled by default (None). The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a :class:`datetime.timedelta`, or a :class:`numpy.timedelta64`.
                            """)
        limit_train_batches = NumberField(value=1.0, checker=number_check(lower=0.0), help="""
                                          How much of training dataset to check (floats = percent, int = num_batches). Default: ``1.0``.
                                          """)
        limit_val_batches = NumberField(value=1.0, checker=number_check(lower=0.0), help="""
                                        How much of validation dataset to check (floats = percent, int = num_batches). Default: ``1.0``.
                                        """)
        limit_test_batches = NumberField(value=1.0, checker=number_check(lower=0.0), help="""
                                         How much of test dataset to check (floats = percent, int = num_batches). Default: ``1.0``.
                                         """)
        limit_predict_batches = NumberField(value=1.0, checker=number_check(lower=0.0), help="""
                                            How much of prediction dataset to check (floats = percent, int = num_batches). Default: ``1.0``.
                                            """)
        overfit_batches = NumberField(value=0.0, checker=number_check(lower=0.0), help="""
                                      Uses this much of training data, and will use the same on val/test/predict. Default: ``0.0``.
                                      """)
        val_check_interval = NumberField(value=1.0, checker=number_check(lower=0.0), help="""
                                         How often to check the validation set. Use float to check within a training epoch, use int to check every n steps (batches). Default: ``1.0``.
                                         """)
        check_val_every_n_epoch = IntField(value=1, checker=int_check(additions=None), help="""
                                           Perform a validation loop every `N` training epochs. If ``None``, validation will be done solely based on the number of training batches, requiring ``val_check_interval`` to be an integer value. Default: ``1``. 
                                           """)
        num_sanity_val_steps = IntField(value=2, checker=int_check(additions=None), help="""
                                        Sanity check runs n validation batches before starting the training routine. Set it to `-1` to run all batches in all validation dataloaders. Default: ``2``.
                                        """)
        log_every_n_steps = IntField(value=50, checker=int_check(additions=None), help="""
                                     How often to log within steps. Default: ``50``.
                                     """)
        enable_checkpointing = BoolField(value=True, help="""
                                         If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`. Default: ``True``.
                                         """)
        enable_progress_bar = BoolField(value=True, help="""
                                        Whether to enable to progress bar by default. Default: ``True``.
                                        """)
        enable_model_summary = BoolField(value=True, help="""
                                         Whether to enable model summarization by default. Default: ``True``.
                                         """)
        accumulate_grad_batches = IntField(value=1, checker=int_check(additions=None), help="""
                                           Accumulates gradients over k batches before stepping the optimizer. Default: 1.
                                           """)
        gradient_clip_val = NumberField(value=5.0, checker=number_check(lower=0.0, additions=None), help="""
                                        The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before. Default: ``5.0``.
                                        """)
        gradient_clip_algorithm = StrField(value="norm", checker=str_check(options=["norm", "value"]), help="""
                                           The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"`` to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will be set to ``"norm"``.
                                           """)
        deterministic = BoolField(value=None, checker=options([True, False, None]), help="""
                                  If ``True``, sets whether PyTorch operations must use deterministic algorithms. Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations that don't support deterministic mode (requires PyTorch 1.11+). If not set, defaults to ``False``. Default: ``None``.
                                  """)
        benchmark = BoolField(value=None, checker=options([True, False, None]), help="""
                              The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to. The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used (``False`` if not manually set). If :paramref:`~lightning.pytorch.trainer.trainer.Trainer.deterministic` is set to ``True``, this will default to ``False``. Override to manually set a different value. Default: ``None``.
                              """)
        inference_mode = BoolField(value=True, checker=options([True, False]), help="""
                                   Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during evaluation (``validate``/``test``/``predict``).
                                   """)
        use_distributed_sampler = BoolField(value=True, checker=options([True, False]), help="""
                                            Whether to wrap the DataLoader's sampler with :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed sampler was already added, Lightning will not replace the existing one. For iterable-style datasets, we don't do this automatically.
                                            """)
        profiler = AnyField(value=None, checker=options([None]), help="""
                            To profile individual steps during training and assist in identifying bottlenecks. Default: ``None``.
                            """)
        detect_anomaly = BoolField(value=False, checker=options([True, False]), help="""
                                   Enable anomaly detection for the autograd engine. Default: ``False``.
                                   """)
        barebones = BoolField(value=False, checker=options([True, False]), help="""
                              Whether to run in "barebones mode", where all features that may impact raw speed are disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training runs. The following features are deactivated: :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_checkpointing`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.logger`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_progress_bar`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.log_every_n_steps`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_model_summary`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.num_sanity_val_steps`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.fast_dev_run`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.detect_anomaly`, :paramref:`~lightning.pytorch.trainer.trainer.Trainer.profiler`, :meth:`~lightning.pytorch.core.module.LightningModule.log`, :meth:`~lightning.pytorch.core.module.LightningModule.log_dict`.
                              """)
        plugins = ListField(value=None, checker=options(additions=None), help="""
                            Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins. Default: ``None``.
                            """)
        sync_batchnorm = BoolField(value=False, checker=options([True, False]), help="""
                                   Synchronize batch norm layers between process groups/whole world. Default: ``False``.
                                   """)
        reload_dataloaders_every_n_epochs = IntField(value=0, checker=int_check(lower=0), help="""
                                                     Set to a non-negative integer to reload dataloaders every n epochs. Default: ``0``.
                                                     """)
        default_root_dir = StrField(value="*@*", checker=str_check(), help="""
                                    Default path for logs and weights when no logger/ckpt_callback passed. Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'.
                                    """)
    config = NestField(value=Config, converter=nest_converter)
    submods = SubModules({}, help="""
                         the callback modules configure
                         """)


class LightningManagerHelper(object):
    """docstring for LightningManagerConfig
    check https://pytorch-lightning.readthedocs.io trainer for paramaters detail
    """

    def __init__(self, _config: LightningTrainerConfig):
        self.config = _config.to_dict()['config']
        self.config['callback'] = self.get_callbacks_config(self.config)

    def get_callbacks_config(self, config: Dict)->List[Dict]:
        """get the configs for callbacks

        Args:
            config: {"config": {"callbacks": ["callback_names"..]}, "callback@callback_names": {config}}

        Returns: 
            configs which name in config['config']['callbacks']

        """
        callback_names = config.get("config", {}).get("callbacks", [])
        callback_configs_list = []
        for callback_name in callback_names:
            callback_config = config.get(f"callback@{callback_name}", {})
            if not callback_config:
                with open(os.path.join(get_root(), f'dlk/configures/core/callbacks/{callback_name}.hjson'), 'r') as f:
                    callback_config = hjson.load(f, object_pairs_hook=dict)
                parser_callback_config = parser.config_parser_register.get('callback')(callback_config).parser_with_check(parser_link=False)
                assert len(parser_callback_config) == 1, f"Don't support multi callback config for one callback."
                callback_config = parser_callback_config[0]
                assert not callback_config.get("_link", {}), f"Callback don't support _link"
            callback_configs_list.append(callback_config)
        return callback_configs_list


@register("trainer", 'basic')
class LightningTrainer(object):
    """pytorch-lightning traning manager
    """

    def __init__(self, config: LightningTrainerConfig, rt_config: Dict):
        super().__init__()
        config_helper = LightningManagerHelper(config).config
        if config_helper['logger']:
            config_helper['logger'] = TensorBoardLogger(save_dir=os.path.join(rt_config["save_dir"], rt_config["name"]), version='')
        if config_helper['callbacks']:
            config_helper['callbacks'] = self.get_callbacks(config_helper['callbacks'], rt_config)
        self.manager = pl.Trainer(**config_helper)

    def get_callbacks(self, callback_configs: List[Dict], rt_config: Dict):
        """init the callbacks and return the callbacks list

        Args:
            callback_configs: the config of every callback
            rt_config: {"save_dir": '..', "name": '..'}

        Returns: 
            all callbacks

        """
        callbacks_list = []

        for callback_config in callback_configs:
            Callback, CallbackConfig = ConfigTool.get_leaf_module(register, config_register, "callback", callback_config)
            callbacks_list.append(Callback(CallbackConfig)(rt_config=rt_config))
        return callbacks_list

    def fit(self, **inputs):
        """fit the model and datamodule to trainer

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns: 
            Undefine

        """
        
        return self.manager.fit(**inputs)

    def predict(self, **inputs):
        """fit the model and datamodule.predict_dataloader to predict

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns: 
            predict list

        """
        return self.manager.predict(**inputs)

    def test(self, **inputs):
        """fit the model and datamodule.test_dataloader to test

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns: 
            Undefine

        """
        return self.manager.test(**inputs)

    def validate(self, **inputs):
        """fit the model and datamodule.validation to validate

        Args:
            **inputs: dict of input, include "model", 'datamodule'

        Returns: 
            Undefine

        """
        return self.manager.validate(**inputs)
