# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from functools import lru_cache
from typing import Callable, Dict, List, Union

import lightning as pl
import torch
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
    asdict,
    cregister,
)

from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


def check_accept_submodule(submodule_config: Dict):
    """check the submodule config is valid
    Args:
        submodule_config:
            the submodule config, it should be a dict, the key is the submodule name
    Returns:
        None
    Raises:
        KeyError: if the submodule name is not accept
    """
    accept_submodule = [
        "model",
        "optimizer",
        "scheduler",
        "loss",
        "postprocessor",
        "adv_method",
    ]
    for key in submodule_config:
        key = key.split("@")[0].split("#")[0].split("-")[0]
        if key not in accept_submodule:
            logger.error(
                f"The submodule name {key} is not accept for default IModel, please use one of {accept_submodule}."
            )
            return False
    return True


@cregister("imodel", "default")
class DefaultIModelConfig(Base):
    """
    The Most General IModel, include the model, optimizer, scheduler, loss, postprocessor
    """

    gather_metrics = BoolField(
        value=True,
        help="when using the distributed training, whether gather all the result to rank 0 to calculate the metrics, if `False` we will only calculate the metrics which on rank 0.",
    )
    validate_names = ListField(
        value=["val"],
        help="the name(s) of the validation datasets(if there are more than one validation dataset, you should provide the names for each one), used in the training process",
    )
    test_names = ListField(
        value=["test"],
        help="the name(s) of the test datasets(if there are more than one test dataset, you should provide the names for each one), used in the training process",
    )

    submodule = SubModule(
        value={},
        suggestions=[
            "model",
            "optimizer",
            "scheduler",
            "loss",
            "postprocessor",
            "adv_method",
        ],
        validator=check_accept_submodule,
    )


@register("imodel", "default")
class DefaultIModel(pl.LightningModule):
    """ """

    def __init__(
        self, config: DefaultIModelConfig, checkpoint=False, rt_config: Dict = {}
    ):
        """init all modules except scheduler which requires the information from datamodule(training steps and every epoch steps)"""
        super().__init__()
        self.config = config
        model_configs = config._get_modules("model")
        assert (
            len(model_configs) == 1
        ), "The model submodule should only have one config"
        self.model = register.get(
            "model", register_module_name(model_configs[0]._module_name)
        )(model_configs[0], checkpoint)

        self.train_rt_config = rt_config
        loss_configs = config._get_modules("loss")
        assert len(loss_configs) == 1, "The loss submodule should only have one config"
        self.calc_loss = register.get(
            "loss", register_module_name(loss_configs[0]._module_name)
        )(loss_configs[0])

        postprocessor_configs = config._get_modules("postprocessor")
        assert (
            len(postprocessor_configs) == 1
        ), "The postprocessor submodule should only have one config"
        self.postprocessor = register.get(
            "postprocessor", register_module_name(postprocessor_configs[0]._module_name)
        )(postprocessor_configs[0])

        adv_method_configs = config._get_modules("adv_method")

        if len(adv_method_configs) == 1:
            self.adv_method = register.get(
                "adv_method", register_module_name(adv_method_configs[0]._module_name)
            )(self.model, adv_method_configs[0])
            self.automatic_optimization = False
        else:
            self.adv_method = None
        self.validation_outputs = {}
        self.test_outputs = {}
        self._reset_output("valid")
        self._reset_output("test")

        self._origin_data = {}
        self.gather_data: Dict = asdict(postprocessor_configs[0].input_map)
        self.gather_data.update(postprocessor_configs[0].predict_extend_return)

    def _reset_output(self, stage: str):
        """reset the validation and test output to empty
        Returns:
            None
        """
        if stage == "valid":
            for i, name in enumerate(self.config.validate_names):
                self.validation_outputs[i] = {"name": name, "outputs": []}
        elif stage == "test":
            for i, name in enumerate(self.config.test_names):
                self.test_outputs[i] = {"name": name, "outputs": []}
        else:
            raise ValueError(f"stage should be valid or test, but got {stage}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            loss, loss_log = self.calc_loss(
                result,
                batch,
                rt_config={
                    "current_step": self.global_step,
                    "current_epoch": self.current_epoch,
                    "total_steps": self.num_training_steps,
                    "total_epochs": self.num_training_epochs,
                },
            )
        log_info = {}
        for key in loss_log:
            log_info[f"train_{key}"] = loss_log[key].unsqueeze(0).detach().cpu()
        self.log_dict(log_info, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
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
        loss, loss_log = self.calc_loss(
            result,
            batch,
            rt_config={  # align with training step
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
            },
        )
        log_info = {}
        for key in loss_log:
            log_info[f"{key}"] = loss_log[key].unsqueeze(0).detach().cpu()
        gather_column = list(self.gather_data.keys())
        return_result = log_info  # this loss will be used in postprocess
        for column in gather_column:
            column = self.gather_data[column]
            if column in result:
                return_result[column] = result[column]
        return_result["_index"] = batch["_index"]

        self.validation_outputs[dataloader_idx]["outputs"].append(return_result)
        return return_result

    def on_validation_epoch_end(self) -> None:
        """Gather the outputs of all node and do postprocess on it.

        The outputs only gather the keys in self.gather_data.keys for postprocess

        Returns:
            None

        """
        if self.config.gather_metrics:
            for index, validation_output in self.validation_outputs.items():
                if self.trainer.world_size > 1:
                    outputs_list = [[] for _ in range(self.trainer.world_size)]
                    torch.distributed.all_gather_object(
                        outputs_list, validation_output["outputs"]
                    )
                    outputs = []
                    for output in outputs_list:
                        outputs.extend(output)
                else:
                    outputs = validation_output["outputs"]
                self.validation_outputs[index]["outputs"] = outputs

        for index, validation_output in self.validation_outputs.items():
            log_dict = self.postprocessor(
                stage="valid",
                list_batch_outputs=validation_output["outputs"],
                origin_data=self._origin_data["valid"],
                rt_config={
                    "current_step": self.global_step,
                    "current_epoch": self.current_epoch,
                    "total_steps": self.num_training_steps,
                    "total_epochs": self.num_training_epochs,
                    "log_name": validation_output["name"],
                },
            )
            self.log_dict(
                log_dict,
                prog_bar=True,
                rank_zero_only=True,
            )
            if self.trainer.loggers and self.train_rt_config["hp_metrics"] in log_dict:
                hp_met = self.train_rt_config["hp_metrics"]
                self.trainer.loggers[0].log_hyperparams(
                    self.train_rt_config["hyper_config"],
                    metrics={hp_met: log_dict[hp_met]},
                )
        self._reset_output("valid")
        return None

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
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
        loss, loss_log = self.calc_loss(
            result,
            batch,
            rt_config={  # align with training step
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
            },
        )
        log_info = {}
        for key in loss_log:
            log_info[f"{key}"] = loss_log[key].unsqueeze(0)
        gather_column = list(self.gather_data.keys())
        return_result = log_info  # this loss will use in postprocess
        for column in gather_column:
            if column in result:
                return_result[column] = result[column]
        return_result["_index"] = batch["_index"]
        self.test_outputs[dataloader_idx]["outputs"].append(return_result)
        return None

    def on_test_epoch_end(self) -> None:
        """Gather the outputs of all node and do postprocess on it.

        Returns:
            None
        """
        if self.config.gather_metrics:
            for index, test_output in self.test_outputs.items():
                if self.trainer.world_size > 1:
                    outputs_list = [[] for _ in range(self.trainer.world_size)]
                    torch.distributed.all_gather_object(
                        outputs_list, test_output["outputs"]
                    )
                    outputs = []
                    for output in outputs_list:
                        outputs.extend(output)
                else:
                    outputs = test_output["outputs"]
                self.test_outputs[index]["outputs"] = outputs

        for index, test_output in self.test_outputs.items():
            log_dict = self.postprocessor(
                stage="test",
                list_batch_outputs=test_output["outputs"],
                origin_data=self._origin_data["test"],
                rt_config={
                    "current_step": self.global_step,
                    "current_epoch": self.current_epoch,
                    "total_steps": self.num_training_steps,
                    "total_epochs": self.num_training_epochs,
                    "log_name": test_output["name"],
                },
            )
            self.log_dict(
                log_dict,
                prog_bar=True,
                rank_zero_only=True,
            )
            if self.trainer.loggers and self.train_rt_config["hp_metrics"] in log_dict:
                hp_met = self.train_rt_config["hp_metrics"]
                self.trainer.loggers[0].log_hyperparams(
                    self.train_rt_config["hyper_config"],
                    metrics={hp_met: log_dict[hp_met]},
                )
        self._reset_output("test")
        return None

    def predict_step(self, batch: Dict, batch_idx: int) -> Dict:
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
        return_result["_index"] = batch["_index"]
        return return_result

    @property
    @lru_cache(maxsize=5)  # the size should always == 1
    def num_training_epochs(self) -> int:
        """Total training epochs inferred from datamodule and devices."""
        return self.trainer.max_epochs

    @property
    @lru_cache(maxsize=2)  # the size should always == 1
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps
        # FIXED: https://github.com/PyTorchLightning/pytorch-lightning/pull/11599
        # NEED TEST
        return int(self.trainer.estimated_stepping_batches)

    def configure_optimizers(self):
        """Configure the optimizer and scheduler"""

        self.calc_loss.update_config(
            rt_config={
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
            }
        )

        optimizer_configs = self.config._get_modules("optimizer")
        assert (
            len(optimizer_configs) == 1
        ), "The optimizer submodule should only have one config"
        optimizer = register.get(
            "optimizer", register_module_name(optimizer_configs[0]._module_name)
        )(self.model, optimizer_configs[0]).get_optimizer()

        scheduler_configs = self.config._get_modules("scheduler")
        assert (
            len(scheduler_configs) == 1
        ), f"The scheduler submodule should only have one config. {scheduler_configs}"
        scheduler = register.get(
            "scheduler", register_module_name(scheduler_configs[0]._module_name)
        )(
            optimizer,
            scheduler_configs[0],
            rt_config={
                "num_training_steps": self.num_training_steps,
                "num_training_epochs": self.num_training_epochs,
            },
        ).get_scheduler()
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
