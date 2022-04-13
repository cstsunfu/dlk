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

import hjson
import os
from typing import Dict, Union, Callable, List, Any
from dlk.utils.parser import BaseConfigParser
from dlk.utils.config import ConfigTool
from dlk.data.datamodules import datamodule_register, datamodule_config_register
from dlk.managers import manager_register, manager_config_register
from dlk.core.imodels import imodel_register, imodel_config_register
import pickle as pkl
import torch
import copy
import uuid
import json
from dlk.utils.logger import Logger

logger = Logger.get_logger()


class Predict(object):
    """Predict

    Config Example:
        >>> {
        >>>     "_focus": {
        >>>     },
        >>>     "_link": {},
        >>>     "_search": {},
        >>>     "config": {
        >>>         "save_dir": "*@*",  # must be provided
        >>>         "data_path": "*@*",  # must be provided
        >>>     },
        >>>     "task": {
        >>>         "_name": task_name
        >>>         ...
        >>>     }
        >>> }
    """
    def __init__(self, config, checkpoint):
        super(Predict, self).__init__()
        if not isinstance(config, dict):
            config = hjson.load(open(config), object_pairs_hook=dict)

        self.focus = config.pop('_focus', {})
        configs = BaseConfigParser(config).parser_with_check()
        assert len(configs) == 1, f"For predict currently the config length must be 1(you cannot use _search in predict)."
        self.config = configs[0]

        self.ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
        config_name = []
        for source, to in self.focus.items():
            config_point = self.config
            trace = source.split('.')
            for t in trace:
                config_point = config_point[t]
            config_name.append(to+str(config_point))
        if config_name:
            name_str = '_'.join(config_name)
        else:
            name_str = self.config['root']['_name']
        self.name_str = name_str

    def trace(self):
        """trace the model to torchscript
        Returns: 
            TODO

        """
        config = self.config['root']
        name = self.name_str
        # get data
        data = self.get_data(config)

        # set datamodule
        datamodule = self.get_datamodule(config, data)

        # init imodel and inject the origin test and valid data
        imodel = self.get_imodel(config, data)

        dataloader = datamodule.train_dataloader()
        for data in dataloader:
            # script = torch.jit.trace(imodel.model, example_inputs=data, strict=False)
            script = torch.jit.trace(imodel.model, example_inputs=data, strict=False)
            # script = torch.jit.script(imodel.model, example_inputs=data, strict=False)
            print(script)
            print(script(data))
            # imodel.model(data)
            break
        logger.error('The trace method is not implement yet.')
        raise NotImplementedError

    def predict(self, data=None, save_condition=False):
        """init the model, datamodule, manager then predict the predict_dataloader

        Args:
            data: if provide will not load from data_path

        Returns: 
            None

        """
        config = copy.deepcopy(self.config['root'])
        name = self.name_str
        # get data
        if not data:
            data = self.get_data(config)

        # set datamodule
        datamodule = self.get_datamodule(config, data)

        # set training manager
        manager = self.get_manager(config, name)

        # init imodel and inject the origin test and valid data
        imodel = self.get_imodel(config, data)

        # start predict
        predict_result = manager.predict(model=imodel, datamodule=datamodule)
        return imodel.postprocessor(stage='predict', list_batch_outputs=predict_result, origin_data=data['predict'], rt_config={}, save_condition=save_condition)

    def get_data(self, config):
        """get the data decided by config

        Args:
            config: {"config": {"data_path": '..'}}

        Returns: 
            loaded data

        """
        
        self.data = pkl.load(open(config['config']['data_path'], 'rb')).get('data', {})
        return self.data

    def get_datamodule(self, config, data):
        """get the datamodule decided by config, and fit the data to datamodule

        Args:
            config: {"task": {"datamodule": '..'}}
            data: {"train": '..', 'valid': '..', ..}

        Returns: 
            datamodule

        """
        DataModule, DataModuleConfig = ConfigTool.get_leaf_module(datamodule_register, datamodule_config_register, 'datamodule', config['task']['datamodule'])
        datamodule = DataModule(DataModuleConfig, data)
        return datamodule

    def get_manager(self, config, name):
        """get the tranin/predict manager decided by config

        Args:
            config: {"task": {"manager": '..'}, "config": {"save_dir"}}
            name: the predict progress name

        Returns: 
            manager

        """
        Manager, ManagerConfig = ConfigTool.get_leaf_module(manager_register, manager_config_register, 'manager', config.get('task').get('manager'))
        manager = Manager(ManagerConfig, rt_config={"save_dir": config.get('config').get("save_dir"), "name": name})
        return manager

    def get_imodel(self, config, data):
        """get the imodel decided by config

        Args:
            config: {"task": {"imodel": '..'}}
            data: {"train": '..', 'valid': '..', ..}

        Returns: 
            imodel

        """
        IModel, IModelConfig = ConfigTool.get_leaf_module(imodel_register, imodel_config_register, 'imodel', config.get('task').get('imodel'))
        imodel = IModel(IModelConfig, checkpoint=True)
        imodel.load_state_dict(self.ckpt['state_dict'])
        imodel.eval()
        return imodel
