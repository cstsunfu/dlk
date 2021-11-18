import torch.nn as nn
from . import manager_register, manager_config_register
from typing import Dict, List
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint
        

@manager_config_register('lightning')
class LightningManagerConfig(object):
    """docstring for LightningManagerConfig
    https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html?highlight=trainer#trainer
    """
    def __init__(self, config):
        super(LightningManagerConfig, self).__init__()
        config = config.get('config')
        # TODO: callback
        self.callbacks =config.get("callbacks", {}), 
        self.logger = config.get("logger", True)
        self.enable_checkpointing = config.get("enable_checkpointing", False) # use checkpoint callback

        self.accelerator = config.get("accelerator", None)

        self.default_root_dir = config.get("default_root_dir", None)
        self.gradient_clip_val = config.get("gradient_clip_val", None)
        self.gradient_clip_algorithm = config.get("gradient_clip_algorithm", None) # TODO: ? default = 'norm', can select 'norm' or 'value' 
        self.num_nodes = config.get("num_nodes", 1)
        self.num_processes = config.get("num_processes", 1)
        self.devices = config.get("devices", None)
        self.gpus = config.get("gpus", None)
        self.auto_select_gpus = config.get("auto_select_gpus", False)
        self.tpu_cores = config.get("tpu_cores", None)
        self.ipus = config.get("ipus", None)
        self.log_gpu_memory = config.get("log_gpu_memory", None)
        self.enable_progress_bar = config.get("enable_progress_bar", True)
        self.overfit_batches = config.get("overfit_batches", 0.0)
        self.track_grad_norm = config.get("track_grad_norm", - 1)
        self.check_val_every_n_epoch = config.get("check_val_every_n_epoch", 1)
        self.fast_dev_run = config.get("fast_dev_run", False)
        self.accumulate_grad_batches = config.get("accumulate_grad_batches", 1)
        self.max_epochs = config.get("max_epochs", None)
        self.min_epochs = config.get("min_epochs", None)
        self.max_steps = config.get("max_steps", -1)
        self.min_steps = config.get("min_steps", None)
        self.max_time = config.get("max_time", None)
        self.limit_train_batches = config.get("limit_train_batches", 1.0)
        self.limit_val_batches = config.get("limit_val_batches", 1.0)
        self.limit_test_batches = config.get("limit_test_batches", 1.0)
        self.limit_predict_batches = config.get("limit_predict_batches", 1.0)
        self.val_check_interval = config.get("val_check_interval", 1.0)
        self.log_every_n_steps = config.get("log_every_n_steps", 50)
        self.strategy = config.get("strategy", 'ddp') # use ddp as default
        self.sync_batchnorm = config.get("sync_batchnorm", False)
        self.precision = config.get("precision", 32)
        self.enable_model_summary = config.get("enable_model_summary", True)
        self.weights_summary = config.get("weights_summary", 'top')
        self.weights_save_path = config.get("weights_save_path", None)
        self.num_sanity_val_steps = config.get("num_sanity_val_steps", 2)
        self.resume_from_checkpoint = config.get("resume_from_checkpoint", None)
        self.profiler = config.get("profiler", None) # 'simple', 'pytorch', etc.
        self.benchmark = config.get("benchmark", False)
        self.deterministic = config.get("deterministic", False)
        self.reload_dataloaders_every_n_epochs = config.get("reload_dataloaders_every_n_epochs", 0)
        self.auto_lr_find = config.get("auto_lr_find", False)
        self.replace_sampler_ddp = config.get("replace_sampler_ddp", True)
        self.detect_anomaly = config.get("detect_anomaly", False)
        self.auto_scale_batch_size = config.get("auto_scale_batch_size", False)
        self.plugins = config.get("plugins", None) # TODO: add plugins from parser plugins config
        self.amp_backend = config.get("amp_backend", 'native') # pytorch>1.6
        self.amp_level = config.get("amp_level", None) # if not set amp_backend to "apex", don't need change this

        self.move_metrics_to_cpu = config.get("move_metrics_to_cpu", False)
        self.multiple_trainloader_mode = config.get("multiple_trainloader_mode", 'max_size_cycle')
        self.stochastic_weight_avg = config.get("stochastic_weight_avg", False)
        self.terminate_on_nan = config.get("terminate_on_nan", None)


@manager_register('lightning')
class LightningManager(object):
    """
    """

    def __init__(self, config: LightningManagerConfig, rt_config: Dict):
        super().__init__()
        config.callbacks = self.get_callbacks(config.callbacks, rt_config)
        self.manager = pl.Trainer(**config.__dict__)

    def get_callbacks(self, config, rt_config):
        """TODO: Docstring for get_callbacks.
        :returns: TODO

        """
        return []

    def fit(self, **inputs):
        return self.manager.fit(**inputs)

    def predict(self, **inputs):
        return self.manager.predict(**inputs)

    def test(self, **inputs):
        return self.manager.test(**inputs)

    def validate(self, **inputs):
        return self.manager.validate(**inputs)
