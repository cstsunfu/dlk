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

"""postprocessors"""

import importlib
import os
from typing import Callable, Dict, Type, List, Union
from dlk.utils.register import Register
from dlk.utils.config import BaseConfig
import torch
import pandas as pd
import abc

class IPostProcessorConfig(BaseConfig):
    """docstring for PostProcessorConfigBase"""
    def __init__(self, config):
        super(IPostProcessorConfig, self).__init__(config)
        self.config = config.get('config', {})

    @property
    def predict_extend_return(self):
        """save the extend data in predict

        Returns: 
            predict_extend_return
        """
        return self.config.get('predict_extend_return', {})

    @property
    def input_map(self):
        """required the output of model process content name map

        Returns: 
            input_map
        """
        return self.config.get("input_map", {})

    @property
    def origin_input_map(self):
        """required the origin data(before pass to datamodule) column name map

        Returns: 
            origin_input_map
        """
        return self.config.get("origin_input_map", {})


class IPostProcessor(metaclass=abc.ABCMeta):
    """docstring for IPostProcessor"""

    def loss_name_map(self, stage)->str:
        """get the stage loss name

        Args:
            stage: valid, train or test

        Returns: 
            loss_name

        """
        map = {
            "valid": 'val',
            'train': 'train',
            "test": "test",
        }
        return map.get(stage, stage)

    def gather_predict_extend_data(self, input_data: Dict, i: int, predict_extend_return: Dict):
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

    def average_loss(self, list_batch_outputs: List[Dict])->float:
        """average all the loss of the list_batches

        Args:
            list_batch_outputs: a list of outputs

        Returns: 
            average_loss

        """
        sum_loss = 0
        for batch_output in list_batch_outputs:
            sum_loss += batch_output.get('loss', 0)
        return sum_loss / len(list_batch_outputs)

    @abc.abstractmethod
    def do_predict(self, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict)->List:
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
    def do_calc_metrics(self, predicts: List, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict)->Dict:
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
    def do_save(self, predicts: List, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict, save_condition: bool=False):
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
    def without_ground_truth_stage(self)->set:
        """there is not groud truth in the returned stage

        Returns: 
            without_ground_truth_stage

        """
        return {'predict', 'online'}

    def process(self, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict, save_condition: bool=False)->Union[Dict, List]:
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
            log_info[f'{self.loss_name_map(stage)}_loss'] = average_loss
        predicts = self.do_predict(stage, list_batch_outputs, origin_data, rt_config)
        if stage not in self.without_ground_truth_stage:
            log_info.update(self.do_calc_metrics(predicts, stage, list_batch_outputs, origin_data, rt_config))
        if stage == 'online':
            return predicts
        if stage == 'predict':
            if save_condition:
                self.do_save(predicts, stage, list_batch_outputs, origin_data, rt_config, save_condition=True)
            return predicts
        else:
            if save_condition:
                self.do_save(predicts, stage, list_batch_outputs, origin_data, rt_config, save_condition=True)
            else:
                self.do_save(predicts, stage, list_batch_outputs, origin_data, rt_config, save_condition=False)
            return log_info

    def __call__(self, stage, list_batch_outputs, origin_data, rt_config, save_condition=False):
        """the same as self.process
        """
        return self.process(stage, list_batch_outputs, origin_data, rt_config, save_condition)


postprocessor_config_register = Register('PostProcessor config register')
postprocessor_register = Register("PostProcessor register")


def import_postprocessors(postprocessors_dir, namespace):
    for file in os.listdir(postprocessors_dir):
        path = os.path.join(postprocessors_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            # and not (file.endswith("subpostprocessors") and os.path.isdir(path))
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            postprocessor_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + postprocessor_name)


# automatically import any Python files in the models directory
postprocessors_dir = os.path.dirname(__file__)
import_postprocessors(postprocessors_dir, "dlk.data.postprocessors")
