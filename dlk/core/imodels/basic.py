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

from logging import PercentStyle
import hjson
from typing import Dict, Union, Callable, List
import torch
from dlk.core.models import model_register, model_config_register
from dlk.core.optimizers import optimizer_register, optimizer_config_register
from dlk.core.schedulers import scheduler_register, scheduler_config_register
from dlk.core.losses import loss_register, loss_config_register
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register
from dlk.utils.config import BaseConfig, ConfigTool
from . import imodel_config_register, imodel_register, GatherOutputMixin
from dlk.utils.logger import Logger
from functools import lru_cache
import copy
logger = Logger.get_logger()

import pytorch_lightning as pl

@imodel_config_register('basic')
class BasicIModelConfig(BaseConfig):
    """ basic imodel config will provide all the config for model/optimizer/loss/scheduler/postprocess
    """
    def __init__(self, config: Dict):
        super(BasicIModelConfig, self).__init__(config)
        self.model, self.model_config = self.get_model(config.pop("model"))
        self.loss, self.loss_config = self.get_loss(config.pop("loss"))

        self.optimizer, self.optimizer_config = self.get_optimizer(config.pop("optimizer", 'adamw'))

        self.scheduler, self.scheduler_config = self.get_scheduler(config.pop("scheduler", "basic"))
        self.postprocess, self.postprocess_config = self.get_postprocessor(config.pop("postprocessor", 'identity'))

        self.config = config.pop('config', {})

    def get_postprocessor(self, config: Dict):
        """Use config to init the postprocessor

        Args:
            config: postprocess config

        Returns: 
            PostProcess, PostProcessConfig

        """
        return  ConfigTool.get_leaf_module(postprocessor_register, postprocessor_config_register, 'postprocessor', config)

    def get_model(self, config: Dict):
        """Use config to init the model

        Args:
            config: model config

        Returns: 
            Model, ModelConfig

        """
        return ConfigTool.get_leaf_module(model_register, model_config_register, "model", config)

    def get_loss(self, config: Dict):
        """Use config to init the loss

        Args:
            config: loss config

        Returns: 
            Loss, LossConfig

        """
        return ConfigTool.get_leaf_module(loss_register, loss_config_register, "loss", config)

    def get_optimizer(self, config: Dict):
        """Use config to init the optimizer

        Args:
            config: optimizer config

        Returns: 
            Optimizer, OptimizerConfig

        """
        return ConfigTool.get_leaf_module(optimizer_register, optimizer_config_register, "optimizer", config)

    def get_scheduler(self, config: Dict):
        """Use config to init the scheduler

        Args:
            config: scheduler config

        Returns: 
            Scheduler, SchedulerConfig

        """
        return ConfigTool.get_leaf_module(scheduler_register, scheduler_config_register, "scheduler", config)


@imodel_register("basic")
class BasicIModel(pl.LightningModule, GatherOutputMixin):
    """
    """
    def __init__(self, config: BasicIModelConfig, checkpoint=False):
        """ init all modules except scheduler which requires the information from datamodule(training steps and every epoch steps)
        """
        super().__init__()
        self.config = config  # scheduler will init in configure_optimizers, because it will use the datamodule info

        self.model = config.model(config.model_config, checkpoint)


        self.calc_loss = config.loss(config.loss_config)
        self.get_optimizer = config.optimizer(model=self.model, config=config.optimizer_config)

        self._origin_valid_data = None
        self._origin_test_data = None
        self.postprocessor = config.postprocess(config.postprocess_config)
        self.gather_data: Dict = copy.deepcopy(config.postprocess_config.input_map)
        self.gather_data.update(config.postprocess_config.predict_extend_return)

    def get_progress_bar_dict(self):
        """rewrite the prograss_bar_dict, remove the 'v_num' which we don't need

        Returns: 
            progress_bar dict

        """
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            the outputs

        """
        return self.model(inputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """do training_step on a mini batch

        Args:
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns: 
            the outputs

        """
        result = self.model.training_step(batch)
        loss = self.calc_loss(result, batch, rt_config={
            "current_step": self.global_step,
            "current_epoch": self.current_epoch,
            "total_steps": self.num_training_steps,
            "total_epochs": self.num_training_epochs
        })
        self.log_dict({"train_loss": loss.unsqueeze(0)}, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int)->Dict[str, torch.Tensor]:
        """do validation on a mini batch

        The outputs only gather the keys in self.gather_data.keys for postprocess
        Args:
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns: 
            the outputs

        """
        result = self.model.validation_step(batch)
        loss = self.calc_loss(result, batch, rt_config={  # align with training step
            "current_step": self.global_step,
            "current_epoch": self.current_epoch,
            "total_steps": self.num_training_steps,
            "total_epochs": self.num_training_epochs
        })
        gather_column = list(self.gather_data.keys())
        return_result = {"loss": loss.unsqueeze(0)} # this loss will be used in postprocess
        for column in gather_column:
            column = self.gather_data[column]
            if column in result:
                return_result[column] = result[column]
        return_result['_index'] = batch['_index']
        return return_result

    def validation_epoch_end(self, outputs: List[Dict])->List[Dict]:
        """Gather the outputs of all node and do postprocess on it.

        The outputs only gather the keys in self.gather_data.keys for postprocess
        Args:
            outputs: current node returnd output list

        Returns: 
            all node outputs

        """
        outputs = self.gather_outputs(outputs)

        self.log_dict(
            self.postprocessor(stage='valid', list_batch_outputs=outputs, origin_data=self._origin_valid_data,
                rt_config={
                    "current_step": self.global_step,
                    "current_epoch": self.current_epoch,
                    "total_steps": self.num_training_steps,
                    "total_epochs": self.num_training_epochs
                }),
            prog_bar=True, rank_zero_only=True)
        return outputs

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int)->Dict:
        """do test on a mini batch

        The outputs only gather the keys in self.gather_data.keys for postprocess
        Args:
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns: 
            the outputs

        """
        result = self.model.test_step(batch)
        loss = self.calc_loss(result, batch, rt_config={  # align with training step
            "current_step": self.global_step,
            "current_epoch": self.current_epoch,
            "total_steps": self.num_training_steps,
            "total_epochs": self.num_training_epochs
        })
        gather_column = list(self.gather_data.keys())
        return_result = {"loss": loss.unsqueeze(0)} # this loss will use in postprocess
        for column in gather_column:
            if column in result:
                return_result[column] = result[column]
        return_result['_index'] = batch['_index']
        return return_result

    def test_epoch_end(self, outputs: List[Dict])->List[Dict]:
        """Gather the outputs of all node and do postprocess on it.

        Args:
            outputs: current node returnd output list

        Returns: 
            all node outputs

        """
        outputs = self.gather_outputs(outputs)

        self.log_dict(
            self.postprocessor(stage='test', list_batch_outputs=outputs, origin_data=self._origin_test_data,
                rt_config={
                    "current_step": self.global_step,
                    "current_epoch": self.current_epoch,
                    "total_steps": self.num_training_steps,
                    "total_epochs": self.num_training_epochs
                }),
            prog_bar=True, rank_zero_only=True)
        return outputs

    def predict_step(self, batch: Dict, batch_idx: int)->Dict:
        """do predict on a mini batch

        Args:
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns: 
            the outputs

        """
        result = self.model.predict_step(batch)
        gather_column = list(self.gather_data.keys())
        return_result = {}
        for column in gather_column:
            column = self.gather_data[column]
            if column in result:
                return_result[column] = result[column]
        return_result['_index'] = batch['_index']
        return return_result

    @property
    @lru_cache(maxsize=5) # the size should always == 1
    def num_training_epochs(self) -> int:
        """Total training epochs inferred from datamodule and devices.
        """
        return self.trainer.max_epochs

    @property
    @lru_cache(maxsize=5) # the size should always == 1
    def epoch_training_steps(self) -> int:
        """every epoch training steps inferred from datamodule and devices.
        """
        if self.trainer.datamodule.train_dataloader() is None:
            batches = 0
        else:
            batches = len(self.trainer.datamodule.train_dataloader())
        return batches

    @property
    @lru_cache(maxsize=5) # the size should always == 1
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices.
        """
         # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449 should check update
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        if self.trainer.datamodule.train_dataloader() is None:
            batches = 0
        else:
            batches = len(self.trainer.datamodule.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum + (1 if batches%effective_accum else 0)) * self.trainer.max_epochs

    def configure_optimizers(self):
        """Configure the optimizer and scheduler
        """

        self.calc_loss.update_config(rt_config={
            "total_steps": self.num_training_steps,
            "total_epochs": self.num_training_epochs
        })

        optimizer = self.get_optimizer()
        if "num_training_steps" in self.config.scheduler_config.__dict__:
            self.config.scheduler_config.num_training_steps = self.num_training_steps
        if "epoch_training_steps" in self.config.scheduler_config.__dict__:
            self.config.scheduler_config.epoch_training_steps = self.epoch_training_steps
        if "num_training_epochs" in self.config.scheduler_config.__dict__:
            self.config.scheduler_config.num_training_epochs = self.num_training_epochs
        # self.config.scheduler_config.last_epoch = -1
        scheduler = self.config.scheduler(optimizer, self.config.scheduler_config)()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
