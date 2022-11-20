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
from . import manager_register, manager_config_register
from typing import Dict, List
import hjson
import pytorch_lightning as pl
from dlk.utils.config import BaseConfig, ConfigTool
from dlk.utils.get_root import get_root
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from dlk.core.callbacks import callback_register, callback_config_register
from dlk.utils.io import open
import dlk.utils.parser as parser
from pytorch_lightning.loggers import TensorBoardLogger


@manager_config_register('lightning')
class LightningManagerConfig(BaseConfig):
    default_config = {
        "_name": "lightning",
        "config":{
            "callbacks": ["checkpoint@val_loss"],
            "logger": True,
            "enable_checkpointing": True,
            "accelerator": None,
            "default_root_dir": None,
            "gradient_clip_val": 5.0,   # gradient clip > 5.0
            "gradient_clip_algorithm": None,
            "num_nodes": 1,
            "devices": None,
            "auto_select_gpus": False,
            "ipus": None,
            "log_gpu_memory": None,
            "enable_progress_bar": True,
            "overfit_batches": "0.0", # eval it to check it is float or int
            "track_grad_norm": -1,
            "check_val_every_n_epoch": 1,
            "fast_dev_run": False,
            "accumulate_grad_batches": 1,
            "max_epochs": 100,
            "min_epochs": None,
            "max_steps": -1,
            "min_steps": None,
            "max_time": None,
            "limit_train_batches": "1.0", # eval it to check it is float or int
            "limit_val_batches": "1.0", # eval it to check it is float or int
            "limit_test_batches": "1.0", # eval it to check it is float or int
            "limit_predict_batches": "1.0", # eval it to check it is float or int
            "val_check_interval": "1.0", # eval it to check it is float or int
            "log_every_n_steps": 50,
            "strategy": 'ddp',
            "sync_batchnorm": False,
            "precision": 32,
            "enable_model_summary": True,
            "weights_summary": 'top',
            "weights_save_path": None,
            "num_sanity_val_steps": 2,
            "resume_from_checkpoint": None,
            "profiler": None,
            "benchmark": False,
            "deterministic": False,
            "reload_dataloaders_every_n_epochs": 0,
            "auto_lr_find": False,
            "replace_sampler_ddp": True,
            "detect_anomaly": False,
            "auto_scale_batch_size": False,
            "plugins": None,
            "amp_backend": 'native',
            "amp_level": None,
            "move_metrics_to_cpu": False,
            "multiple_trainloader_mode": 'max_size_cycle',
            "stochastic_weight_avg": False,
            "terminate_on_nan": None,
        },
    }
    """docstring for LightningManagerConfig
    check https://pytorch-lightning.readthedocs.io trainer for paramaters detail
    """

    def __init__(self, config):
        super(LightningManagerConfig, self).__init__(config)

        manager_config = config.get('config')
        self.callbacks = self.get_callbacks_config(config)  # this is callback config, should be initialized Callback in LightningManager
        self.logger = manager_config["logger"] # True

        self.enable_checkpointing = manager_config["enable_checkpointing"] # True  use checkpoint callbac
        self.accelerator = manager_config["accelerator"] # None
        self.default_root_dir = manager_config["default_root_dir"] # None
        self.gradient_clip_val = manager_config["gradient_clip_val"] # None
        self.gradient_clip_algorithm = manager_config["gradient_clip_algorithm"] # None TODO: ? default = 'norm', can select 'norm' or 'value
        self.num_nodes = manager_config["num_nodes"] # 1
        self.devices = manager_config["devices"] # None
        self.auto_select_gpus = manager_config["auto_select_gpus"] # False
        self.ipus = manager_config["ipus"] # None
        self.log_gpu_memory = manager_config["log_gpu_memory"] # None
        self.enable_progress_bar = manager_config["enable_progress_bar"] # True
        self.overfit_batches = eval(manager_config["overfit_batches"])
        self.track_grad_norm = manager_config["track_grad_norm"] # -1
        self.check_val_every_n_epoch = manager_config["check_val_every_n_epoch"] # 1
        self.fast_dev_run = manager_config["fast_dev_run"] # False
        self.accumulate_grad_batches = manager_config["accumulate_grad_batches"] # 1
        self.max_epochs = manager_config["max_epochs"] # None
        self.min_epochs = manager_config["min_epochs"] # None
        self.max_steps = manager_config["max_steps"] # -1
        self.min_steps = manager_config["min_steps"] # None
        self.max_time = manager_config["max_time"] # None

        self.limit_train_batches = eval(manager_config["limit_train_batches"])
        self.limit_val_batches = eval(manager_config["limit_val_batches"])
        self.limit_test_batches = eval(manager_config["limit_test_batches"])
        self.limit_predict_batches = eval(manager_config["limit_predict_batches"])
        self.val_check_interval = eval(manager_config["val_check_interval"])
        self.log_every_n_steps = manager_config["log_every_n_steps"] # 50
        self.strategy = manager_config["strategy"] # 'ddp' use ddp as default
        self.sync_batchnorm = manager_config["sync_batchnorm"] # False
        self.precision = manager_config["precision"] # 32
        self.enable_model_summary = manager_config["enable_model_summary"] # True
        self.weights_summary = manager_config["weights_summary"] # 'top'
        self.weights_save_path = manager_config["weights_save_path"] # None
        self.num_sanity_val_steps = manager_config["num_sanity_val_steps"] # 2
        self.resume_from_checkpoint = manager_config["resume_from_checkpoint"] # None
        self.profiler = manager_config["profiler"] # None  'simple', 'pytorch', etc
        self.benchmark = manager_config["benchmark"] # False
        self.deterministic = manager_config["deterministic"] # False
        self.reload_dataloaders_every_n_epochs = manager_config["reload_dataloaders_every_n_epochs"] # 0
        self.auto_lr_find = manager_config["auto_lr_find"] # False
        self.replace_sampler_ddp = manager_config["replace_sampler_ddp"] # True
        self.detect_anomaly = manager_config["detect_anomaly"] # False
        self.auto_scale_batch_size = manager_config["auto_scale_batch_size"] # False
        self.plugins = manager_config["plugins"] # None TODO: add plugins from parser plugins config
        self.amp_backend = manager_config["amp_backend"] # 'native'  pytorch>1.6
        self.amp_level = manager_config["amp_level"] # None  if not set amp_backend to "apex", don't need change this

        self.move_metrics_to_cpu = manager_config["move_metrics_to_cpu"] # False
        self.multiple_trainloader_mode = manager_config["multiple_trainloader_mode"] # 'max_size_cycle'
        self.stochastic_weight_avg = manager_config["stochastic_weight_avg"] # False
        self.terminate_on_nan = manager_config["terminate_on_nan"] # None
        self.post_check(manager_config, used=[
            "callbacks",
            "logger",
            "enable_checkpointing",
            "accelerator",
            "default_root_dir",
            "gradient_clip_val",
            "gradient_clip_algorithm",
            "num_nodes",
            "devices",
            "auto_select_gpus",
            "ipus",
            "log_gpu_memory",
            "enable_progress_bar",
            "overfit_batches",
            "track_grad_norm",
            "check_val_every_n_epoch",
            "fast_dev_run",
            "accumulate_grad_batches",
            "max_epochs",
            "min_epochs",
            "max_steps",
            "min_steps",
            "max_time",
            "limit_train_batches",
            "limit_val_batches",
            "limit_test_batches",
            "limit_predict_batches",
            "val_check_interval",
            "log_every_n_steps",
            "strategy",
            "sync_batchnorm",
            "precision",
            "enable_model_summary",
            "weights_summary",
            "weights_save_path",
            "num_sanity_val_steps",
            "resume_from_checkpoint",
            "profiler",
            "benchmark",
            "deterministic",
            "reload_dataloaders_every_n_epochs",
            "auto_lr_find",
            "replace_sampler_ddp",
            "detect_anomaly",
            "auto_scale_batch_size",
            "plugins",
            "amp_backend",
            "amp_level",
            "move_metrics_to_cpu",
            "multiple_trainloader_mode",
            "stochastic_weight_avg",
            "terminate_on_nan",
        ])

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


@manager_register('lightning')
class LightningManager(object):
    """pytorch-lightning traning manager
    """

    def __init__(self, config: LightningManagerConfig, rt_config: Dict):
        super().__init__()

        if config.logger:
            config.logger = TensorBoardLogger(save_dir=os.path.join(rt_config["save_dir"], rt_config["name"]), version='')
        if config.callbacks:
            config.callbacks = self.get_callbacks(config.callbacks, rt_config)
        config.__dict__.pop('_name')
        self.manager = pl.Trainer(**config.__dict__)

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
            Callback, CallbackConfig = ConfigTool.get_leaf_module(callback_register, callback_config_register, "callback", callback_config)
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
