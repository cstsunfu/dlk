"""
Provide BaseConfig which provide the basic method for configs, and ConfigTool a general config(dict) process tool
"""
from typing import Any, Dict, Union, Callable, List, Tuple, Type
import json
import copy
import os
from dlk.utils.logger import Logger
from dlk.utils.register import Register

logger = Logger.get_logger()

class BaseConfig(object):
    """docstring for BaseLayerConfig"""
    def __init__(self, config: Dict):
        super(BaseConfig, self).__init__()
        self._name = config.pop('_name')

    def post_check(self, config, used=[]):
        """check all the params are useful
        :config: TODO
        :returns: TODO
        """
        def rec_pop(cur_node, trace):
            """TODO: Docstring for rec_pop.
            """
            if len(trace) > 1:
                rec_pop(cur_node[trace[0]], trace[1:])
            if len(trace) == 1:
                if cur_node.get(trace[0], {}) == {}:
                    cur_node.pop(trace[0])

        config = copy.deepcopy(config)
        parant_traces = set()
        for key in used:
            sp_key = key.split('.')
            parant_traces.add(tuple(sp_key[:-1]))
            cur_root = config
            for key in sp_key[:-1]:
                cur_root = cur_root[key]
            cur_root.pop(sp_key[-1], None)
        for trace in parant_traces:
            rec_pop(config, trace)

        if config:
            logger.warning(f"In module '{self._name}', there are some params not be used: {config}")


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
                if "_name" in _base[item] and "_name" in _new[item]:
                    if _base[item]['_name'] != _new[item]['_name']:
                        logger.warning(f"The Higher Config for {_new[item]['_name']} Coverd the Base {_base[item]['_name']} ")
                        _base[item] = _new[item]

                        continue
                ConfigTool._inplace_update_dict(_base[item], _new[item])
            else:
                raise AttributeError("The base config and update config is not match. base: {}, new: {}. ".format(_base, _new))

    @staticmethod
    def do_update_config(config: dict, update_config: dict={}) ->Dict:
        """use update_config update the config

        :config: will updated config
        :returns: updated config

        """
        # BUG ?: if the config._name != update_config._name, should use the update_config conver the config wholely
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
                },
                "predict": "train",
                "online": ["train",
                {"vocab": "new_label_vocab"}
                ]
            }
            config.get_config['predict'] == config['predict'] == config['train']
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
