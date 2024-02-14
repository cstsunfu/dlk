# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import abc
import importlib
import json
import os
from typing import Callable, Dict, List, Type, TypeVar, Union

import pandas as pd
import pyarrow.parquet as pq
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
    dataclass,
)

from dlk.utils.import_module import import_module_dir


@dataclass
class BasePostProcessorConfig(Base):
    """the base postprocessor"""

    meta_dir = StrField(
        value="data/meta_data",
        help="the save dir of the meta info",
    )
    save_root_path = StrField(
        value="data",
        help="the root path of the save data, default relative to the current path",
    )
    save_dir = DictField(
        value={"valid": "valid_output", "test": "test_output"},
        help="the save path of the data, relative to the save_root_path",
    )
    start_save_step = IntField(
        value=0, minimum=-1, help="the start save step, -1 means the last step"
    )
    start_save_epoch = IntField(
        value=-1, minimum=-1, help="the start save epoch, -1 means the last epoch"
    )
    predict_extend_return = DictField(value={}, help="the extend return of predict")


class BasePostProcessor(object):
    """the base postprocessor"""

    def __init__(self, config):
        super(BasePostProcessor, self).__init__()

    def loss_name_map(self, stage) -> str:
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

    def average_loss(self, list_batch_outputs: List[Dict]) -> Dict[str, float]:
        """average all the loss of the list_batches

        Args:
            list_batch_outputs: a list of outputs

        Returns:
            average_loss

        """
        loss_names = []
        average_losses = {}
        batch_num = len(list_batch_outputs)
        if not batch_num:
            return average_losses
        for key in list_batch_outputs[0]:
            if key.endswith("_loss") or key == "loss":
                loss_names.append(key)
                average_losses[key] = 0
        for batch_output in list_batch_outputs:
            for name in loss_names:
                average_losses[name] = average_losses[name] + batch_output.get(name, 0)
        average_losses = {
            key: value / batch_num for key, value in average_losses.items()
        }
        return average_losses

    @abc.abstractmethod
    def do_predict(
        self,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
    ) -> List:
        """Process the model predict to human readable format

        Args:
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
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
        raise NotImplementedError

    @abc.abstractmethod
    def do_calc_metrics(
        self,
        predicts: List,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
    ) -> Dict:
        """calc the scores use the predicts or list_batch_outputs

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
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

    @abc.abstractmethod
    def do_save(
        self,
        predicts: List,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
        save_condition: bool = False,
    ):
        """save the predict when save_condition==True

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
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
        raise NotImplementedError

    @property
    def without_ground_truth_stage(self) -> set:
        """there is not groud truth in the returned stage

        Returns:
            without_ground_truth_stage

        """
        return {"predict", "online"}

    def process(
        self,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
        save_condition: bool = False,
    ) -> Union[Dict, List]:
        """PostProcess entry

        Args:
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
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
        log_info = {}
        if stage not in self.without_ground_truth_stage:
            average_loss = self.average_loss(list_batch_outputs=list_batch_outputs)
            for name in average_loss:
                log_info[f"{self.loss_name_map(stage)}_{name}"] = average_loss[name]
        predicts = self.do_predict(stage, list_batch_outputs, origin_data, rt_config)
        if stage not in self.without_ground_truth_stage:
            log_info.update(
                self.do_calc_metrics(
                    predicts, stage, list_batch_outputs, origin_data, rt_config
                )
            )

        if stage == "online":
            return predicts
        if stage == "predict":
            if save_condition:
                self.do_save(
                    predicts,
                    stage,
                    list_batch_outputs,
                    origin_data,
                    rt_config,
                    save_condition=True,
                )
            return predicts
        else:
            if save_condition:
                self.do_save(
                    predicts,
                    stage,
                    list_batch_outputs,
                    origin_data,
                    rt_config,
                    save_condition=True,
                )
            else:
                self.do_save(
                    predicts,
                    stage,
                    list_batch_outputs,
                    origin_data,
                    rt_config,
                    save_condition=False,
                )
            return log_info

    def __call__(
        self, stage, list_batch_outputs, origin_data, rt_config, save_condition=False
    ):
        """the same as self.process"""
        return self.process(
            stage, list_batch_outputs, origin_data, rt_config, save_condition
        )


postprocessor_dir = os.path.dirname(__file__)
import_module_dir(postprocessor_dir, "dlk.data.postprocessor")
