# Copyright cstsunfu. All rights reserved.
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
from typing import Dict, Union, Callable, List
import torch
from dlk.utils.config import BaseConfig, ConfigTool
from . import GatherOutputMixin
from dlk.utils.logger import Logger
from functools import lru_cache
import copy
from dlk import register, config_register
from dlk.core.schedulers import BaseSchedulerConfig
from dlk.utils.config import define, float_check, int_check, str_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, SubModules
logger = Logger.get_logger()

import lightning as pl


@config_register("imodel", 'basic')
@define
class BasicIModelConfig(BaseConfig):
    name = NameField(value="basic", file=__file__, help="register module name")
    @define
    class Config:
        gather_metrics = BoolField(value=True, help="when using the distributed training, whether gather all the result to rank 0 to calculate the metrics, if `False` we will only calculate the metrics which on rank 0.")
        validate_names = ListField(value=["val"], help="the name(s) of the validation datasets(if there are more than one validation dataset, you should provide the names for each one), used in the training process")
        test_names = ListField(value=["test"], help="the name(s) of the test datasets(if there are more than one test dataset, you should provide the names for each one), used in the training process")

    config = NestField(value=Config, converter=nest_converter)
    submods = SubModules({ 
                          "model": "*@*",
                          "loss": "*@*",
                          "optimizer": {
                              "base": "adamw@bias_nodecay",
                              },
                          "scheduler": {
                              "base": "linear_warmup"
                              },
                          "postprocessor": {
                              "base": "identity"
                              },
                          "adv_method": {
                              "base": "identity"
                              },
                          })

class BasicIModelHelper(object):
    """ basic imodel config will provide all the config for model/optimizer/loss/scheduler/postprocess
    """
    def __init__(self, config: BasicIModelConfig):
        super(BasicIModelHelper, self).__init__()
        config_dict = config.to_dict()
        self.model, self.model_config = self.get_leaf_module(config_dict.pop("model"), "model")
        self.loss, self.loss_config = self.get_leaf_module(config_dict.pop("loss"), "loss")

        self.optimizer, self.optimizer_config = self.get_leaf_module(config_dict.pop("optimizer"), "optimizer")
        if config_dict['adv_method']['name'] == 'identity':
            self.adv_method = None
        else:
            self.adv_method, self.adv_method_config = self.get_leaf_module(config_dict.pop("adv_method"), "adv_method")

        self.scheduler, self.scheduler_config = self.get_leaf_module(config_dict.pop("scheduler"), "scheduler")
        self.scheduler_config: BaseSchedulerConfig
        self.postprocess, self.postprocess_config = self.get_leaf_module(config_dict.pop("postprocessor"), "postprocessor")

        self.gather_metrics = config.config.gather_metrics
        self.validate_names = config.config.validate_names
        self.test_names = config.config.test_names

    def get_leaf_module(self, config: Dict, module_type_name: str):
        """Use config to init the module

        Args:
            config: module config

        Returns: 
            Moule, ModuleConfig

        """
        return  ConfigTool.get_leaf_module(register, config_register, module_type_name, config)


@register("imodel", "basic")
class BasicIModel(pl.LightningModule, GatherOutputMixin):
    """
    """
    def __init__(self, config: BasicIModelConfig, checkpoint=False):
        """ init all modules except scheduler which requires the information from datamodule(training steps and every epoch steps)
        """
        super().__init__()
        self.config = BasicIModelHelper(config)  # scheduler will init in configure_optimizers, because it will use the datamodule info

        self.model = self.config.model(self.config.model_config, checkpoint)


        self.calc_loss = self.config.loss(self.config.loss_config)
        self.get_optimizer = self.config.optimizer(model=self.model, config=self.config.optimizer_config)
        if self.config.adv_method:
            self.adv_method = self.config.adv_method(model=self.model, config=self.config.adv_method_config)
            self.automatic_optimization = False
        else:
            self.adv_method = None
        self.validation_outputs = {}
        self.test_outputs = {}
        self._reset_output("valid")
        self._reset_output("test")

        self._origin_valid_data = {}
        self._origin_test_data = {}
        self.postprocessor = self.config.postprocess(self.config.postprocess_config)
        self.gather_data: Dict = copy.deepcopy(self.config.postprocess_config.config.input_map)
        self.gather_data.update(self.config.postprocess_config.config.predict_extend_return)

    def _reset_output(self, stage: str):
        """reset the validation and test output to empty
        Returns: 
            None
        """
        if stage == 'valid':
            for i, name in enumerate(self.config.validate_names):
                self.validation_outputs[i] = {"name": name, "outputs": []}
        elif stage == 'test':
            for i, name in enumerate(self.config.test_names):
                self.test_outputs[i] = {"name": name, "outputs": []}
        else:
            raise ValueError(f"stage should be valid or test, but got {stage}")

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
        if self.adv_method:
            loss, loss_log = self.adv_method.training_step(self, batch, batch_idx)
        else:
            result = self.model.training_step(batch)
            loss, loss_log = self.calc_loss(result, batch, rt_config={
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs
            })
        log_info = {}
        for key in loss_log:
            log_info[f"train_{key}"] = loss_log[key].unsqueeze(0).detach().cpu()
        self.log_dict(log_info, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int=0)->Dict[str, torch.Tensor]:
        """do validation on a mini batch

        The outputs only gather the keys in self.gather_data.keys for postprocess
        Args:
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch
            dataloader_idx: the index of the multi dataloaders

        Returns: 
            the outputs

        """
        result = self.model.validation_step(batch)
        loss, loss_log = self.calc_loss(result, batch, rt_config={  # align with training step
            "current_step": self.global_step,
            "current_epoch": self.current_epoch,
            "total_steps": self.num_training_steps,
            "total_epochs": self.num_training_epochs
        })
        log_info = {}
        for key in loss_log:
            log_info[f"{key}"] = loss_log[key].unsqueeze(0).detach().cpu()
        gather_column = list(self.gather_data.keys())
        return_result = log_info # this loss will be used in postprocess
        for column in gather_column:
            column = self.gather_data[column]
            if column in result:
                return_result[column] = result[column]
        return_result['_index'] = batch['_index']

        self.validation_outputs[dataloader_idx]["outputs"].append(return_result)
        return return_result

    def on_validation_epoch_end(self)->None:
        """Gather the outputs of all node and do postprocess on it.

        The outputs only gather the keys in self.gather_data.keys for postprocess

        Returns: 
            None

        """
# self.logger.log_hyperparams({"total_epochs": 8, "optimizer": "SGD", 'val_hyper_loss': torch.tensor(random.random(), requires_grad=True), 'val_hyper_acc': torch.tensor(random.random(), requires_grad=True)})
        if self.config.gather_metrics:
            for index, validation_output in self.validation_outputs.items():
                outputs_list = [[] for _ in range(self.trainer.world_size)]
                torch.distributed.all_gather_object(outputs_list, validation_output["outputs"])
                outputs = []
                for output in outputs_list:
                    outputs.extend(output)
                self.validation_outputs[index]["outputs"] = outputs

        for index, validation_output in self.validation_outputs.items():
            self.log_dict(
                self.postprocessor(stage='valid', list_batch_outputs=validation_output['outputs'], origin_data=self._origin_valid_data[validation_output["name"]],
                    rt_config={
                        "current_step": self.global_step,
                        "current_epoch": self.current_epoch,
                        "total_steps": self.num_training_steps,
                        "total_epochs": self.num_training_epochs,
                        "log_name": validation_output["name"]
                    }),
                prog_bar=True, rank_zero_only=True)
        self._reset_output("valid")
        return None

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int=0)->None:
        """do test on a mini batch

        The outputs only gather the keys in self.gather_data.keys for postprocess
        Args:
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch
            dataloader_idx: the index of the multi dataloaders

        Returns: 
            None

        """
        result = self.model.test_step(batch)
        loss, loss_log = self.calc_loss(result, batch, rt_config={  # align with training step
            "current_step": self.global_step,
            "current_epoch": self.current_epoch,
            "total_steps": self.num_training_steps,
            "total_epochs": self.num_training_epochs
        })
        log_info = {}
        for key in loss_log:
            log_info[f"{key}"] = loss_log[key].unsqueeze(0)
        gather_column = list(self.gather_data.keys())
        return_result = log_info # this loss will use in postprocess
        for column in gather_column:
            if column in result:
                return_result[column] = result[column]
        return_result['_index'] = batch['_index']
        self.test_outputs[dataloader_idx]["outputs"].append(return_result)
        return None

    def test_epoch_end(self)->None:
        """Gather the outputs of all node and do postprocess on it.

        Returns: 
            None
        """
        if self.config.gather_metrics:
            for index, test_output in self.test_outputs.items():
                outputs_list = [[] for _ in range(self.trainer.world_size)]
                torch.distributed.all_gather_object(outputs_list, test_output["outputs"])
                outputs = []
                for output in outputs_list:
                    outputs.extend(output)
                self.test_outputs[index]["outputs"] = outputs

        for index, test_output in self.test_outputs.items():
            self.log_dict(
                self.postprocessor(stage='test', list_batch_outputs=test_output['outputs'], origin_data=self._origin_test_data[test_output['name']],
                    rt_config={
                        "current_step": self.global_step,
                        "current_epoch": self.current_epoch,
                        "total_steps": self.num_training_steps,
                        "total_epochs": self.num_training_epochs,
                        "log_name": test_output["name"]
                    }),
                prog_bar=True, rank_zero_only=True)
        self._reset_output("test")
        return None

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
    @lru_cache(maxsize=2) # the size should always == 1
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices.
        """
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps
        # FIXED: https://github.com/PyTorchLightning/pytorch-lightning/pull/11599
        # NEED TEST
        return int(self.trainer.estimated_stepping_batches)

    def configure_optimizers(self):
        """Configure the optimizer and scheduler
        """

        self.calc_loss.update_config(rt_config={
            "total_steps": self.num_training_steps,
            "total_epochs": self.num_training_epochs
        })

        optimizer = self.get_optimizer()
        scheduler = self.config.scheduler(optimizer, self.config.scheduler_config, rt_config={
            "num_training_steps": self.num_training_steps,
            "num_training_epochs": self.num_training_epochs
            })()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
