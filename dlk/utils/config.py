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
    """BaseConfig provide the basic function for all config"""
    def __init__(self, config: Dict):
        super(BaseConfig, self).__init__()
        self._name = config.pop('_name')

    def post_check(self, config, used=None):
        """check all the paras in config is used

        Args:
            config: paras
            used: used paras

        Returns: 
            None

        Raises: 
            logger.warning("Unused")

        """
        if not used:
            used = []
        def rec_pop(cur_node, trace):
            """recursive pop the node if the node == {} and the node path is in trace
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
    def _inplace_update_dict(_base: Dict, _new: Dict):
        """use the _new dict inplace update the _base dict, recursively

        if the _base['_name'] != _new["_name"], we will use _new cover the _base and logger a warning
        otherwise, use _new update the _base recursively

        Args:
            _base: will be updated dict
            _new: use _new update _base

        Returns: 
            None

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
    def do_update_config(config: dict, update_config: dict=None) ->Dict:
        """use the update_config dict update the config dict, recursively

        see ConfigTool._inplace_update_dict

        Args:
            config: will be updated dict
            update_confg: config: use _new update _base

        Returns: 
            updated_config

        """
        if not update_config:
            update_config = {}
        # BUG ?: if the config._name != update_config._name, should use the update_config conver the config wholely
        config = copy.deepcopy(config)
        ConfigTool._inplace_update_dict(config, update_config)
        return config

    @staticmethod
    def get_leaf_module(module_register: Register, module_config_register: Register, module_name: str, config: Dict) -> Tuple[Any, object]:
        """get the module from module_register and module_config from module_config_register which name=module_name

        Args:
            module_register: register for module which has 'module_name'
            module_config_register: config register for config which has 'module_name'
            module_name: the module name which we want to get from register

        Returns: 
            module(which name is module_name), module_config(which name is module_name)

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
        """get the stage_config for special stage in provide config

        it means the config of this stage equals to config[stage]
        return config[config[stage]]

        Config Example:
            >>> config = {
            >>>     "train":{ //train、predict、online stage config,  using '&' split all stages
            >>>         "data_pair": {
            >>>             "label": "label_id"
            >>>         },
            >>>         "data_set": {                   // for different stage, this processor will process different part of data
            >>>             "train": ['train', 'dev'],
            >>>             "predict": ['predict'],
            >>>             "online": ['online']
            >>>         },
            >>>         "vocab": "label_vocab", // usually provided by the "token_gather" module
            >>>     },
            >>>     "predict": "train",
            >>>     "online": ["train",
            >>>     {"vocab": "new_label_vocab"}
            >>>     ]
            >>> }
            >>> config.get_config['predict'] == config['predict'] == config['train']

        Args:
            stage: the stage, like 'train', 'predict', etc.
            config: the base config which has different stage config

        Returns: 
            stage_config
        """
        config = config['config']
        stage_config = config.get(stage, {})
        if isinstance(stage_config, str):
            stage_config = config.get(stage_config, {})
        elif isinstance(stage_config, list):
            assert len(stage_config) == 2
            assert isinstance(stage_config[0], str)
            assert isinstance(stage_config[1], dict)
            base_config = config.get(stage_config[0], {})
            stage_config = ConfigTool.do_update_config(base_config, stage_config[1])
        return stage_config
