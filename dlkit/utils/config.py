from typing import Any, Dict, Union, Callable, List, Tuple, Type
import json
import copy
import os
from dlkit.utils.logger import get_logger
from dlkit.utils.register import Register


class ConfigTool(object):
    """
    This Class is not be used as much as I design.
    """

    @staticmethod
    def _inplace_update_dict(_base, _new):
        """TODO: Docstring for _inplace_update_dict.
        :returns: TODO

        """
        for item in _new:
            if (item not in _base) or (not isinstance(_base[item], Dict)):
            # if item not in _base, or _base[item] is not Dict
                _base[item] = _new[item]
            elif isinstance(_base[item], Dict) and isinstance(_new[item], Dict):
                ConfigTool._inplace_update_dict(_base[item], _new[item])
            else:
                print(f"base {_base[item]} type: {type(_base[item])}")
                print(f"new {_new[item]} type: {type(_new[item])}")
                raise AttributeError("The base config and update config is not match. base: {}, new: {}. ".format(_base, _new))

    @staticmethod
    def do_update_config(config: dict, update_config: dict={}) ->Dict:
        """use update_config update the config

        :config: will updated config
        :returns: updated config

        """
        config = copy.deepcopy(config)
        ConfigTool._inplace_update_dict(config, update_config)
        return config

    @staticmethod
    def get_leaf_module(module_register: Register, module_config_register: Register, module_name: str, config: Dict) -> Tuple[Any, object]:
        """get sub module and config from register.
          for model, the leaf like encoder decoder and embedding class and the config of class could get by this mixin
        :module_register: Dict[model_name, Model]
        :module_config_register: Dict[model_config_name, ModelConfig]
        :module_name: for echo the log
        :config: Dict[key, value]
        :returns: tuple(Model, ModelConfig)

        """
        if isinstance(config, str):
            name = config
            extend_config = {}
        else:
            assert isinstance(config, dict), "{} config must be name(str) or config(dict), but you provide {}".format(module_name, config)
            name = config.get('_name', "") # must provide _name_
            extend_config = config
            if not name:
                raise KeyError('You must provide the {} name("name")'.format(module_name))

        module_class, module_config_class =  module_register.get(name), module_config_register.get(name)
        if (not module_class) or not (module_config_class):
            raise KeyError('The {} name {} is not registed.'.format(module_name, config))
        module_config = module_config_class(extend_config)
        return module_class, module_config

    @staticmethod
    def get_config_by_stage(stage:str, config:Dict)->Dict:
        """TODO: if config[stage] is a string, like 'train', 'predict' etc., 
            it means the config of this stage equals to config[stage]
            return config[config[stage]]
            e.g.
            config = {
                "train":{ //train、predict、online stage config,  using '&' split all stages
                    "data_pair": {
                        "label": "label_id"
                    },
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'dev'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "vocab": "label_vocab", // usually provided by the "token_gather" module
                }, //3
                "predict": "train",
                "online": ["train",
                {"vocab": "new_label_vocab"}
                ]
            }
            config.get_config['predict'] == config[config['predict']] == config['train']
        """
        config = config['config']
        stage_config = config.get(stage, {})
        if isinstance(stage_config, str):
            stage_config = config.get(stage_config, {})
        elif isinstance(stage_config, list):
            assert len(stage_config) == 2 
            assert isinstance(stage_config[0], str)
            assert isinstance(stage_config[1], dict)
            stage_config = config.get(stage_config[0], {})
            ConfigTool.do_update_config(stage_config, stage_config[1])
        return stage_config
