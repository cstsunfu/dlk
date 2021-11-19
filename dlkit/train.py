from logging import Manager
import hjson
import os
from typing import Dict, Union, Callable, List, Any
from dlkit.utils.parser import config_parser_register 
from dlkit.utils.config import ConfigTool
from dlkit.datamodules import datamodule_register, datamodule_config_register
from dlkit.managers import manager_register, manager_config_register
from dlkit.imodels import imodel_register, imodel_config_register
from dlkit.postprocessors import postprocessor_register, postprocessor_config_register
import pickle as pkl
import uuid
import json


class Train(object):
    """docstring for Trainer
        {
            "_name": "task_name",
            "_focus": {

            },
            "_link": {},
            "_setting": {
                "save_dir": "*@*",  # must provide
            }
        }
    """
    def __init__(self, config):
        super(Train, self).__init__()
        if not isinstance(config, dict):
            self.config = hjson.load(open(config), object_pairs_hook=dict)
        else:
            self.config = config
        self.focus = self.config.pop('_focus', {})

        self.setting = self.config.pop("_setting", {})

        self.configs = config_parser_register.get('task')(self.config).parser_with_check()

        self.config_names = []
        for possible_config in self.configs:
            config_name = []
            for source, to in self.focus.items():
                config_point = possible_config
                trace = source.split('.')
                for t in trace:
                    config_point = config_point[t]
                config_name.append(to+"="+str(config_point))
            if config_name:
                self.config_names.append('_'.join(config_name))
            else:
                self.config_names.append(uuid.uuid1())

        if len(self.config_names) != len(set(self.config_names)):
            for config, name in zip(self.configs, self.config_names):
                print(f"{name}:\n{json.dumps(config, indent=4)}")
            raise NameError('The config_names is not unique.')

    def run(self):
        """TODO: Docstring for run.
        :returns: TODO
        """
        print(f"You have {len(self.config_names)} training config(s), so they all will be run.")
        for i, (config, name) in enumerate(zip(self.configs, self.config_names)):
            print(f"Runing {i}...")
            self.run_oneturn(config, name)

    def run_oneturn(self, config, name):
        """TODO: Docstring for run_oneturn.
        """
        self.pre_run_hook()

        datamodule = self.get_datamodule(config)
        manager = self.get_manager(config, name)
        imodel = self.get_imodel(config)
        imodel.postprocessor = self.get_postprocessor(config)

        manager.fit(model=imodel, datamodule=datamodule)

    def get_data(self, config):
        """TODO: Docstring for get_data.
        :returns: TODO

        """
        return pkl.load(config.get('data_path'))

    def pre_run_hook(self):
        """TODO: Docstring for pre_run_hook.
        :returns: TODO

        """
        pass

    def get_datamodule(self, config):
        """TODO: Docstring for get_datamodule.

        :config: TODO
        :returns: TODO

        """
        DataModule, DataModuleConfig = ConfigTool.get_leaf_module(datamodule_register, datamodule_config_register, 'datamodule', config.get('datamodule'))
        datamodule = DataModule(DataModuleConfig, self.get_data(config.get('config')))
        return datamodule
        
    def get_manager(self, config, name):
        """TODO: Docstring for get_manager.

        :config: TODO
        :returns: TODO

        """
        Manager, ManagerConfig = ConfigTool.get_leaf_module(manager_register, manager_config_register, 'manager', config.get('manager'))
        manager = Manager(ManagerConfig, rt_config={"save_dir": self.setting.get("save_dir"), "name": name})
        return manager

    def get_postprocessor(self, config):
        """TODO: Docstring for get_postprocessor.

        :config: TODO
        :returns: TODO

        """
        PostProcessor, PostProcessorConfig = ConfigTool.get_leaf_module(postprocessor_register, postprocessor_config_register, 'postprocessor', config.get('postprocessor'))
        return PostProcessor(PostProcessorConfig)
        
    def get_imodel(self, config):
        """TODO: Docstring for get_imodel.

        :config: TODO
        :returns: TODO

        """

        IModel, IModelConfig = ConfigTool.get_leaf_module(imodel_register, imodel_config_register, 'imodel', config.get('imodel'))
        imodel = IModel(IModelConfig)
        return imodel
