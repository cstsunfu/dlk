import torch.nn as nn
from . import manager_register, manager_config_register
from typing import Dict, List
import torch
import pytorch_lightning as pl
from dlkit.utils.config import ConfigTool
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from dlkit.callbacks import callback_register, callback_config_register
from pytorch_lightning.loggers import TensorBoardLogger


@manager_config_register('lightning')
class LightningManagerConfig(object):
    """docstring for LightningManagerConfig
    check
    https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html?highlight=trainer#trainer
    for para details
    """


    def __init__(self, config):
        super(LightningManagerConfig, self).__init__()


        manager_config = config.get('config')
        self.callbacks = self.get_callbacks_config(config)  # this is callback config, should be initialized Callback in LightningManager
        self.logger = manager_config.get("logger", True)

        self.enable_checkpointing = manager_config.get("enable_checkpointing", False) # use checkpoint callback
        self.accelerator = manager_config.get("accelerator", None)
        self.default_root_dir = manager_config.get("default_root_dir", None)
        self.gradient_clip_val = manager_config.get("gradient_clip_val", None)
        self.gradient_clip_algorithm = manager_config.get("gradient_clip_algorithm", None) # TODO: ? default = 'norm', can select 'norm' or 'value'
        self.num_nodes = manager_config.get("num_nodes", 1)
        self.num_processes = manager_config.get("num_processes", 1)
        self.devices = manager_config.get("devices", None)
        self.gpus = manager_config.get("gpus", None)
        self.auto_select_gpus = manager_config.get("auto_select_gpus", False)
        self.tpu_cores = manager_config.get("tpu_cores", None)
        self.ipus = manager_config.get("ipus", None)
        self.log_gpu_memory = manager_config.get("log_gpu_memory", None)
        self.enable_progress_bar = manager_config.get("enable_progress_bar", True)
        self.overfit_batches = manager_config.get("overfit_batches", 0.0)
        self.track_grad_norm = manager_config.get("track_grad_norm", - 1)
        self.check_val_every_n_epoch = manager_config.get("check_val_every_n_epoch", 1)
        self.fast_dev_run = manager_config.get("fast_dev_run", False)
        self.accumulate_grad_batches = manager_config.get("accumulate_grad_batches", 1)
        self.max_epochs = manager_config.get("max_epochs", None)
        self.min_epochs = manager_config.get("min_epochs", None)
        self.max_steps = manager_config.get("max_steps", -1)
        self.min_steps = manager_config.get("min_steps", None)
        self.max_time = manager_config.get("max_time", None)
        self.limit_train_batches = manager_config.get("limit_train_batches", 1.0)
        self.limit_val_batches = manager_config.get("limit_val_batches", 1.0)
        self.limit_test_batches = manager_config.get("limit_test_batches", 1.0)
        self.limit_predict_batches = manager_config.get("limit_predict_batches", 1.0)
        self.val_check_interval = manager_config.get("val_check_interval", 1.0)
        self.log_every_n_steps = manager_config.get("log_every_n_steps", 50)
        self.strategy = manager_config.get("strategy", 'ddp') # use ddp as default
        self.sync_batchnorm = manager_config.get("sync_batchnorm", False)
        self.precision = manager_config.get("precision", 32)
        self.enable_model_summary = manager_config.get("enable_model_summary", True)
        self.weights_summary = manager_config.get("weights_summary", 'top')
        self.weights_save_path = manager_config.get("weights_save_path", None)
        self.num_sanity_val_steps = manager_config.get("num_sanity_val_steps", 2)
        self.resume_from_checkpoint = manager_config.get("resume_from_checkpoint", None)
        self.profiler = manager_config.get("profiler", None) # 'simple', 'pytorch', etc.
        self.benchmark = manager_config.get("benchmark", False)
        self.deterministic = manager_config.get("deterministic", False)
        self.reload_dataloaders_every_n_epochs = manager_config.get("reload_dataloaders_every_n_epochs", 0)
        self.auto_lr_find = manager_config.get("auto_lr_find", False)
        self.replace_sampler_ddp = manager_config.get("replace_sampler_ddp", True)
        self.detect_anomaly = manager_config.get("detect_anomaly", False)
        self.auto_scale_batch_size = manager_config.get("auto_scale_batch_size", False)
        self.plugins = manager_config.get("plugins", None) # TODO: add plugins from parser plugins config
        self.amp_backend = manager_config.get("amp_backend", 'native') # pytorch>1.6
        self.amp_level = manager_config.get("amp_level", None) # if not set amp_backend to "apex", don't need change this

        self.move_metrics_to_cpu = manager_config.get("move_metrics_to_cpu", False)
        self.multiple_trainloader_mode = manager_config.get("multiple_trainloader_mode", 'max_size_cycle')
        self.stochastic_weight_avg = manager_config.get("stochastic_weight_avg", False)
        self.terminate_on_nan = manager_config.get("terminate_on_nan", None)

    def get_callbacks_config(self, config):
        """parser callback and init the callbacks
        :returns: list of callbacks

        """
        callback_names = config.get("config", {}).get("callbacks", [])
        callback_configs_list = []
        for callback_name in callback_names:
            callback_config = config.get(f"callback@{callback_name}")
            assert callback_config, f"You want to use callback '{callback_name}', but not provide the config."
            callback_configs_list.append(callback_config)
        return callback_configs_list


@manager_register('lightning')
class LightningManager(object):
    """
    """

    def __init__(self, config: LightningManagerConfig, rt_config: Dict):
        super().__init__()

        if config.logger:
            config.logger = TensorBoardLogger(save_dir=os.path.join(rt_config["save_dir"], rt_config["name"]))
        if config.callbacks:
            config.callbacks = self.get_callbacks(config.callbacks, rt_config)

        self.manager = pl.Trainer(**config.__dict__)

    def get_callbacks(self, callback_configs, rt_config):
        """parser callback and init the callbacks
        :returns: list of callbacks

        """
        callbacks_list = []

        for callback_config in callback_configs:
            Callback, CallbackConfig = ConfigTool.get_leaf_module(callback_register, callback_config_register, "callback", callback_config)
            callbacks_list.append(Callback(CallbackConfig)(rt_config=rt_config))
        return callbacks_list

    def fit(self, **inputs):
        return self.manager.fit(**inputs)

    def predict(self, **inputs):
        return self.manager.predict(**inputs)

    def test(self, **inputs):
        return self.manager.test(**inputs)

    def validate(self, **inputs):
        return self.manager.validate(**inputs)
