# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
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

from dlk.data.postprocessor import BasePostProcessor
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
        self.postprocessor: BasePostProcessor = register.get(
            "postprocessor", register_module_name(postprocessor_configs[0]._module_name)
        )(postprocessor_configs[0])

        adv_method_configs = config._get_modules("adv_method")

        if len(adv_method_configs) == 1 and adv_method_configs[0].use_adv:
            self.adv_method = register.get(
                "adv_method", register_module_name(adv_method_configs[0]._module_name)
            )(self.model, adv_method_configs[0])
            self.automatic_optimization = False
            assert (
                self.trainer.accumulate_grad_batches == 1
            ), f"currently, do not support accumulate for adv train"
        else:
            self.adv_method = None

        self._origin_data = {}
        self.gather_data: Dict = asdict(postprocessor_configs[0].input_map)
        self.gather_data.update(postprocessor_configs[0].predict_extend_return)

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
    ):
        """do validation on a mini batch

        Args:
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch
            dataloader_idx: the index of the multi dataloaders

        Returns:
            the outputs

        """
        batch_output = self.model.validation_step(batch)
        loss, loss_log = self.calc_loss(
            batch_output,
            batch,
            rt_config={  # align with training step
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
            },
        )
        for key in loss_log:
            batch_output[f"{key}"] = loss_log[key].unsqueeze(0).detach().cpu()
        self.postprocessor.update(
            "valid",
            batch_output,
            origin_data=self._origin_data["valid"],
            rt_config={
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
                "log_dir": self.train_rt_config["log_dir"],
                "name": self.train_rt_config["name"],
            },
            index=dataloader_idx,
        )

        return None

    def on_validation_epoch_end(self) -> None:
        """Gather the outputs of all node and do postprocess on it.

        The outputs only gather the keys in self.gather_data.keys for postprocess

        Returns:
            None

        """
        metrics = self.postprocessor.reduce(
            "valid",
            rt_config={
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
                "log_dir": self.train_rt_config["log_dir"],
                "name": self.train_rt_config["name"],
            },
        )
        for i, metric in metrics.items():
            self.log_dict(
                metric,
                prog_bar=True,
                rank_zero_only=True,
            )
            if self.trainer.loggers and self.train_rt_config["hp_metrics"] in metric:
                hp_met = self.train_rt_config["hp_metrics"]
                self.trainer.loggers[0].log_hyperparams(
                    self.train_rt_config["hyper_config"],
                    metrics={hp_met: metric[hp_met]},
                )
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
        batch_output = self.model.test_step(batch)
        loss, loss_log = self.calc_loss(
            batch_output,
            batch,
            rt_config={  # align with training step
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
            },
        )
        for key in loss_log:
            batch_output[f"{key}"] = loss_log[key].unsqueeze(0).detach().cpu()
        self.postprocessor.update(
            "test",
            batch_output,
            origin_data=self._origin_data["test"],
            rt_config={
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
                "log_dir": self.train_rt_config["log_dir"],
                "name": self.train_rt_config["name"],
            },
            index=dataloader_idx,
        )

        return None

    def on_test_epoch_end(self) -> None:
        """Gather the outputs of all node and do postprocess on it.

        Returns:
            None
        """
        metrics = self.postprocessor.reduce(
            "test",
            rt_config={
                "current_step": self.global_step,
                "current_epoch": self.current_epoch,
                "total_steps": self.num_training_steps,
                "total_epochs": self.num_training_epochs,
                "log_dir": self.train_rt_config["log_dir"],
                "name": self.train_rt_config["name"],
            },
        )
        for index, metric in metrics.items():
            self.log_dict(
                metric,
                prog_bar=True,
                rank_zero_only=True,
            )
            if self.trainer.loggers and self.train_rt_config["hp_metrics"] in metric:
                hp_met = self.train_rt_config["hp_metrics"]
                self.trainer.loggers[0].log_hyperparams(
                    self.train_rt_config["hyper_config"],
                    metric={hp_met: metric[hp_met]},
                )
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
    def num_training_epochs(self) -> int:
        """Total training epochs inferred from datamodule and devices."""
        return self.trainer.max_epochs

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps
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
                "interval": scheduler_configs[0].interval,
                "frequency": 1,
            },
        }
