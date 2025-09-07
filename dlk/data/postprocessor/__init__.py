# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import abc
import importlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
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
    dataclass,
)
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

from dlk.utils.import_module import import_module_dir

logger = logging.getLogger(__name__)


@dataclass
class BasePostProcessorConfig(Base):
    """the base postprocessor"""

    meta_dir = StrField(
        value="data/meta_data",
        help="the save dir of the meta info",
    )
    save_root_path = StrField(
        value=None,
        additions=[None],
        help="the root path of the save data, default save to the log dir",
    )
    save_dir = DictField(
        value={
            "valid": "valid_output",
            "test": "test_output",
            "predict": "predict_output",
        },
        help="the save path of the data, relative to the save_root_path",
    )
    start_save_step = IntField(
        value=0, minimum=-1, help="the start save step, -1 means the last step"
    )
    start_save_epoch = IntField(
        value=-1, minimum=-1, help="the start save epoch, -1 means the last epoch"
    )
    predict_extend_return = DictField(value={}, help="the extend return of predict")


class PostInfoCollection(Metric):
    full_state_update = False

    def __init__(self, postprocessor: "BasePostProcessor", dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.postprocessor = postprocessor
        self.loss_logs: List[Dict[str, float]] = []
        self.predicts_list: List[Dict[str, Any]] = []
        self.add_state("loss_logs", default=[], dist_reduce_fx=None)
        self.add_state("predicts_list", default=[], dist_reduce_fx=None)

    def _prepare_loss(self, batch_output, stage):
        cur_loss = {}
        for key in batch_output:
            if key.endswith("_loss") or key == "loss":
                cur_loss[f"{stage}_{key}"] = batch_output[key].detach().cpu().item()
        return cur_loss

    def update(self, stage, batch_output, origin_data, rt_config):
        loss_log = self._prepare_loss(batch_output, stage)
        predict_info = self.postprocessor.wrap_predict_one_batch(
            stage, batch_output, origin_data, rt_config
        )
        self.predicts_list.append(predict_info)
        self.loss_logs.append(loss_log)

    def _average_loss(self, loss_logs: List[Dict]) -> Dict[str, float]:
        """average all the loss of the list_batches

        Args:
            loss_logs: a list of loss_log

        Returns:
            average_loss

        """
        loss_names = []
        average_losses = {}
        batch_num = len(loss_logs)
        if not batch_num:
            return average_losses
        for key in loss_logs[0]:
            if key.endswith("_loss") or key == "loss":
                loss_names.append(key)
                average_losses[key] = 0
        for batch_output in loss_logs:
            for name in loss_names:
                average_losses[name] = average_losses[name] + batch_output.get(name, 0)
        average_losses = {
            key: value / batch_num for key, value in average_losses.items()
        }
        return average_losses

    def compute(
        self, stage, rt_config, save_condition: bool = False
    ) -> Dict[str, torch.Tensor]:
        all_predicts = []
        for predicts in self.predicts_list:
            all_predicts.extend(predicts)

        log_dict = self.postprocessor.do_calc_metrics(
            stage=stage,
            predicts=all_predicts,
            rt_config=rt_config,
        )

        if stage not in self.postprocessor.without_ground_truth_stage:
            average_loss = self._average_loss(loss_logs=self.loss_logs)
            for name in average_loss:
                log_dict[
                    f"{self.postprocessor.loss_name_map(stage)}_{name}"
                ] = average_loss[name]

        self.postprocessor.do_save(
            predicts=self.predicts_list,
            stage=stage,
            rt_config=rt_config,
            save_condition=save_condition,
        )
        return log_dict

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Optional[Any] = None,
    ) -> None:
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)

        if not (dist.is_available() and dist.is_initialized()):
            return

        world_size = dist.get_world_size(group=process_group)

        for state_name, reduce_fn in self._reductions.items():
            # dist_reduce_fx=None and state is a list
            if reduce_fn is None:
                state_value = getattr(self, state_name)
                if isinstance(state_value, list):
                    gathered_data = [None for _ in range(world_size)]

                    dist.all_gather_object(
                        gathered_data, state_value, group=process_group
                    )
                    flattened_data = []
                    for data in gathered_data:
                        if data is not None:
                            flattened_data.extend(data)

                    setattr(self, state_name, flattened_data)


class BasePostProcessor(object):
    """the base postprocessor"""

    def __init__(self, config: BasePostProcessorConfig):
        super(BasePostProcessor, self).__init__()
        self.config = config
        self._info_collections: Dict[str, Dict[int, PostInfoCollection]] = {}

    def loss_name_map(self, stage: str) -> str:
        """get the stage loss name

        Args:
            stage: valid, train or test

        Returns:
            loss_name

        """
        map = {
            "valid": "val",
            "train": "train",
            "test": "test",
        }
        return map.get(stage, stage)

    def gather_predict_extend_data(
        self, input_data: Dict, i: int, predict_extend_return: Dict
    ):
        """gather the data register in `predict_extend_return`
        Args:
            input_data:
                the model output
            i:
                the index is i
            predict_extend_return:
                the name map which will be reserved
        Returns:
            a dict of data in input_data which is register in predict_extend_return
        """
        result = {}
        for key, name in predict_extend_return.items():
            data = input_data[name][i]
            if torch.is_tensor(data):
                data = data.detach().tolist()
            result[key] = data
        return result

    def update(
        self,
        stage: str,
        batch_output: Dict,
        origin_data: List,
        rt_config: Dict,
        index: int,
    ):
        """Process the model predict to human readable format

        Args:
            stage: train/test/etc.
            batch_output: the model output
            origin_data: the origin data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns:
            all predicts

        """
        if stage not in self._info_collections:
            self._info_collections[stage] = {}
        if index not in (self._info_collections[stage]):
            self._info_collections[stage][index] = PostInfoCollection(
                postprocessor=self, dist_sync_on_step=True
            )

        self._info_collections[stage][index].update(
            stage=stage,
            batch_output=batch_output,
            origin_data=origin_data,
            rt_config=rt_config,
        )

    def wrap_predict_one_batch(
        self, stage, batch_output: Dict, origin_data: List, rt_config
    ):
        """prepare the predict one batch for no online/serve stage"""
        raise NotImplementedError

    def predict_one_batch(
        self, stage, batch_output: Dict, origin_data: List, rt_config
    ) -> List:
        """Process the model predict to human readable format for one batch
        Args:
            stage: train/test/etc.
            batch_output: a dict of outputs
            origin_data: the origin data, there are some data not be able to convert to tensor
        Returns:
            the predicts of one batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def do_calc_metrics(
        self,
        predicts: List,
        stage: str,
        rt_config: Dict,
    ) -> Dict:
        """calc the scores use the predicts or list_batch_outputs

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            rt_config:
                >>>
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns:
            the named scores

        """
        raise NotImplementedError

    def do_save(
        self,
        predicts: List,
        stage: str,
        rt_config: Dict,
        save_condition: bool = False,
    ):
        """save the predict when save_condition==True

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }
            save_condition: True for save, False for depend on rt_config

        Returns:
            None

        """
        if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
            self.config.start_save_step = rt_config.get("total_steps", 0) - 1
            self.config.start_save_epoch = rt_config.get("total_epochs", 0) - 1
        if not save_condition and (
            rt_config["current_step"] >= self.config.start_save_step
            or rt_config["current_epoch"] >= self.config.start_save_epoch
        ):
            save_condition = True
        if save_condition:
            if self.config.save_root_path:
                save_path = os.path.join(
                    self.config.save_root_path, self.config.save_dir.get(stage, "")
                )
            else:
                save_path = os.path.join(
                    rt_config["log_dir"],
                    rt_config["name"],
                    self.config.save_dir.get(stage, ""),
                )
            if "current_step" in rt_config:
                save_file = os.path.join(
                    save_path, f"step_{str(rt_config['current_step'])}_predict.json"
                )
            else:
                save_file = os.path.join(save_path, "predict.json")
            logger.info(f"Save the {stage} predict data at {save_file}")
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(predicts, f, indent=4, ensure_ascii=False)

    @property
    def without_ground_truth_stage(self) -> set:
        """there is not groud truth in the returned stage

        Returns:
            without_ground_truth_stage

        """
        return {"predict", "online"}

    def reduce(
        self,
        stage: str,
        rt_config: Dict,
        save_condition: bool = False,
    ):
        """PostProcess entry

        Args:
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }
            save_condition: if save_condition is True, will force save the predict on all stage except online

        Returns:
            the log_info(metrics) or the stage is "online" return the predicts

        """
        metrics = {}
        for index, info_collection in self._info_collections.get(stage, {}).items():
            metrics[index] = info_collection.compute(
                stage=stage,
                rt_config=rt_config,
                save_condition=save_condition,
            )
            info_collection.reset()

        return metrics


postprocessor_dir = os.path.dirname(__file__)
import_module_dir(postprocessor_dir, "dlk.data.postprocessor")
