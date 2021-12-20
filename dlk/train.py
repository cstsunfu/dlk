import hjson
import os
from typing import Dict, Union, Callable, List, Any
from dlk.utils.parser import BaseConfigParser
from dlk.utils.config import ConfigTool
from dlk.data.datamodules import datamodule_register, datamodule_config_register
from dlk.managers import manager_register, manager_config_register
from dlk.core.imodels import imodel_register, imodel_config_register
import pickle as pkl
import uuid
import json
from dlk.utils.logger import Logger
import logging

logger = Logger.get_logger()


class Train(object):
    """docstring for Trainer
        {
            "_focus": {

            },
            "_link": {},
            "_search": {},
            "config": {
                "save_dir": "*@*",  # must be provided
                "data_path": "*@*",  # must be provided
            },
            "task": {
                "_name": task_name
                ...
            }
        }
    """
    def __init__(self, config: Union[str, Dict], ckpt: str=""):
        super(Train, self).__init__()
        if not isinstance(config, dict):
            config = hjson.load(open(config), object_pairs_hook=dict)

        self.ckpt = ckpt
        self.focus = config.pop('_focus', {})
        self.configs = BaseConfigParser(config).parser_with_check()
        if self.ckpt:
            assert len(self.configs) == 1, f"Reuse the checkpoint(ckpt is not none), you must provide the (only one) config which generate the checkpoint."

        self.config_names = []
        for possible_config in self.configs:
            config_name = []
            for source, to in self.focus.items():
                config_point = possible_config
                trace = source.split('.')
                for t in trace:
                    config_point = config_point[t]
                config_name.append(to+str(config_point))
            if config_name:
                self.config_names.append('_'.join(config_name))
            else:
                self.config_names.append(possible_config['root']['_name'])

        if len(self.config_names) != len(set(self.config_names)):
            for config, name in zip(self.configs, self.config_names):
                logger.info(f"{name}:\n{json.dumps(config, indent=4)}")
            raise NameError('The config_names is not unique.')

    def run(self):
        """TODO: Docstring for run.
        :returns: TODO
        """
        logger.info(f"You have {len(self.config_names)} training config(s), they all will be run.")
        for i, (config, name) in enumerate(zip(self.configs, self.config_names)):
            logger.info(f"Runing the {i}th {name}...")
            self.run_oneturn(config, name)

    def dump_config(self, config, name):
        """TODO: Docstring for dump_config.

        :config: TODO
        :returns: TODO
        """
        log_path = os.path.join(config.get('config').get('save_dir'), name)
        os.makedirs(log_path, exist_ok=True)
        json.dump({"root":config, "_focus": self.focus}, open(os.path.join(config.get('config').get('save_dir'), name, "config.json"), 'w'), ensure_ascii=False, indent=4)
        Logger.init_file_logger("log.txt", log_path)

    def run_oneturn(self, config, name):
        """TODO: Docstring for run_oneturn.
        """

        config = config['root']
        # save configure
        self.dump_config(config, name)

        # get data
        data = self.get_data(config)

        # set datamodule
        datamodule = self.get_datamodule(config, data)

        # set training manager
        manager = self.get_manager(config, name)

        # init imodel and inject the origin test and valid data
        imodel = self.get_imodel(config, data)

        # start training
        manager.fit(model=imodel, datamodule=datamodule)
        manager.test(model=imodel, datamodule=datamodule)

    def get_data(self, config):
        """TODO: Docstring for get_data.
        :returns: TODO

        """
        self.data = pkl.load(open(config['config']['data_path'], 'rb')).get('data', {})
        return self.data

    def get_datamodule(self, config, data):
        """TODO: Docstring for get_datamodule.

        :config: TODO
        :returns: TODO

        """
        DataModule, DataModuleConfig = ConfigTool.get_leaf_module(datamodule_register, datamodule_config_register, 'datamodule', config['task']['datamodule'])
        datamodule = DataModule(DataModuleConfig, data)
        return datamodule

    def get_manager(self, config, name):
        """TODO: Docstring for get_manager.

        :config: TODO
        :returns: TODO

        """
        Manager, ManagerConfig = ConfigTool.get_leaf_module(manager_register, manager_config_register, 'manager', config.get('task').get('manager'))
        manager = Manager(ManagerConfig, rt_config={"save_dir": config.get('config').get("save_dir"), "name": name})
        return manager

    def get_imodel(self, config, data):
        """TODO: Docstring for get_imodel.

        :config: TODO
        :returns: TODO

        """
        IModel, IModelConfig = ConfigTool.get_leaf_module(imodel_register, imodel_config_register, 'imodel', config.get('task').get('imodel'))
        imodel = IModel(IModelConfig)
        if self.ckpt:
            logger.info(f"reuse the checkpoint at {self.ckpt}")
            imodel.load_from_checkpoint(self.ckpt)
        if 'valid' in data:
            imodel._origin_valid_data = data['valid']

        if 'test' in data:
            imodel._origin_test_data = data['test']

        return imodel
