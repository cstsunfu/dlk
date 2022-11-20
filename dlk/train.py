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
from dlk.utils.io import open
import json
from dlk.utils.logger import Logger

logger = Logger.get_logger()


class Train(object):
    """Trainer

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
    def __init__(self, config: Union[str, Dict], ckpt: str = "", state_dict_only=False):
        super(Train, self).__init__()
        if not isinstance(config, dict):
            with open(config, 'r') as f:
                config = hjson.load(f, object_pairs_hook=dict)

        self.ckpt = ckpt
        self.state_dict_only = state_dict_only
        self.focus = config.pop('_focus', {})
        self.configs = BaseConfigParser(config).parser_with_check()
        if self.ckpt:
            assert len(
                self.configs
            ) == 1, f"Reuse the checkpoint(ckpt is not none), you must provide the (only one) config which generate the checkpoint."

        self.config_names = []
        for possible_config in self.configs:
            config_name = []
            for source, to in self.focus.items():
                config_point = possible_config
                trace = source.split('.')
                for t in trace:
                    config_point = config_point[t]
                config_name.append(to + str(config_point))
            if config_name:
                self.config_names.append('_'.join(config_name))
            else:
                self.config_names.append(possible_config['root']['_name'])

        if len(self.config_names) != len(set(self.config_names)):
            for config, name in zip(self.configs, self.config_names):
                logger.info(
                    f"{name}:\n{json.dumps(config, indent=4, ensure_ascii=False)}"
                )
            raise NameError('The config_names is not unique.')

    def run(self):
        """run for all configs

        Returns: 
            None

        """
        logger.info(
            f"You have {len(self.config_names)} training config(s), they all will be run."
        )
        for i, (config, name) in enumerate(zip(self.configs,
                                               self.config_names)):
            logger.info(f"Runing the {i}th {name}...")
            self.run_oneturn(config, name)

    def dump_config(self, config: Dict, name: str):
        """dump the config and change the log file path to config['config']['save_dir']+name

        Args:
            config: {"config": {"save_dir": '..'}}
            name: config name

        Returns: 
            None

        """
        log_path = os.path.join(config.get('config').get('save_dir'), name)
        with open(os.path.join(config.get('config').get('save_dir'), name, "config.json"), 'w') as f:
            json.dump(
                {
                    "root": config,
                    "_focus": self.focus
                },
                f,
                ensure_ascii=False,
                indent=4
            )
        Logger.init_file_logger("log.txt", log_path)

    def run_oneturn(self, config, name):
        """run this config

        Args:
            config: {"root": '...'}
            name: config name

        Returns: 
            None

        """

        config = config['root']
        # save configure
        self.dump_config(config, name)

        # get data
        data = self.get_data(config)

        # set training manager
        manager = self.get_manager(config, name)

        # register devices for valid repeat
        # set datamodule
        datamodule = self.get_datamodule(config, data, world_size = manager.manager.world_size)

        # init imodel and inject the origin test and valid data
        imodel = self.get_imodel(config, data)

        # start training
        manager.fit(model=imodel, datamodule=datamodule)
        manager.test(model=imodel, datamodule=datamodule)

    def get_data(self, config):
        """get the data decided by config

        Args:
            config: {"config": {"data_path": '..'}}

        Returns: 
            loaded data

        """
        with open(config['config']['data_path'], 'rb') as f:
            self.data = pkl.load(f).get('data', {})
        return self.data

    def get_datamodule(self, config, data, world_size):
        """get the datamodule decided by config, and fit the data to datamodule

        Args:
            config: {"task": {"datamodule": '..'}}
            data: {"train": '..', 'valid': '..', ..}
            devices: when the devices >1 and repeat_for_valid is True, we will repeat the 

        Returns: 
            datamodule

        """
        DataModule, data_module_config = ConfigTool.get_leaf_module(
            datamodule_register, datamodule_config_register, 'datamodule',
            config['task']['datamodule'])
        data_module_config.world_size = world_size
        datamodule = DataModule(data_module_config, data)
        return datamodule

    def get_manager(self, config, name):
        """get the tranin/predict manager decided by config

        Args:
            config: {"task": {"manager": '..'}, "config": {"save_dir"}}
            name: the predict progress name

        Returns: 
            manager

        """
        Manager, manager_config = ConfigTool.get_leaf_module(
            manager_register, manager_config_register, 'manager',
            config.get('task').get('manager'))
        manager = Manager(manager_config,
                          rt_config={
                              "save_dir": config.get('config').get("save_dir"),
                              "name": name
                          })
        return manager

    def get_imodel(self, config, data):
        """get the imodel decided by config, and inject the origin test and valid data

        Args:
            config: {"task": {"imodel": '..'}}
            data: {"train": '..', 'valid': '..', ..}

        Returns: 
            imodel

        """
        IModel, imodel_config = ConfigTool.get_leaf_module(
            imodel_register, imodel_config_register, 'imodel',
            config.get('task').get('imodel'))
        imodel = IModel(imodel_config)
        if self.ckpt:
            logger.info(f"reuse the checkpoint at {self.ckpt}")
            if self.state_dict_only:
                imodel.model.load_state_dict(torch.load(self.ckpt), strict=False)
            else:
                imodel.load_from_checkpoint(self.ckpt)
        if 'valid' in data:
            imodel._origin_valid_data = data['valid']

        if 'test' in data:
            imodel._origin_test_data = data['test']

        return imodel

