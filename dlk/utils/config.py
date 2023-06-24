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
from collections.abc import Callable
from sys import _getframe as finfo
from attrs import define, field, fields, asdict, fields_dict
from attr._make import _CountingAttr
from typing import Any, List, Set, Type, Dict, TypeVar
import inspect
import json
import time

Dep = TypeVar('Dep')

logger = Logger.get_logger()


def float_check(lower: float=-float('inf'), upper: float=float('inf'), options: List[float]=[], suggestions: List[float]=[], additions: Any=[]):
    """check the value is a float data and is in range (lower, upper)

    Args:
        lower:
            the value should be greater than lower
        upper:
            the value should be less than upper
        options:
            the value should be in options
        additions:
            the values should skip all check, default values is null list, when there is only one addition value, you can set the value itself as additions ranther than a list.
    Returns:
        the value itself
    """
    if not isinstance(additions, List):
        additions = [additions]
    if lower is None:
        lower = -float('inf')
    if upper is None:
        upper = float('inf')

    def _(value: float, options=options, suggestions=suggestions, additions=additions):
        if value in additions:
            return value
        try:
            value = float(value)
        except Exception as e:
            raise e
        if value>=upper or value<=lower:
            raise ValueError(f"Value {value} is not in range ({lower}, {upper})")
        if options:
            if value not in options:
                raise ValueError(f"Value {value} is not in {options}")
        return value
    return _


def number_check(lower: float=-float('inf'), upper: float=float('inf'), options: List[float]=[], suggestions: List[float]=[], additions: Any=[]):
    """check the value is a float data and is in range (lower, upper)

    Args:
        lower:
            the value should be greater than lower
        upper:
            the value should be less than upper
        options:
            the value should be in options
        additions:
            the values should skip all check, default values is null list, when there is only one addition value, you can set the value itself as additions ranther than a list.
    Returns:
        the value itself
    """
    if not isinstance(additions, List):
        additions = [additions]
    if lower is None:
        lower = -float('inf')
    if upper is None:
        upper = float('inf')

    def _(value: float, options=options, suggestions=suggestions, additions=additions):
        if value in additions:
            return value
        assert isinstance(value, (float, int))
        if value>=upper or value<=lower:
            raise ValueError(f"Value {value} is not in range ({lower}, {upper})")
        if options:
            if value not in options:
                raise ValueError(f"Value {value} is not in {options}")
        return value
    return _


def int_check(lower: int=-int(1e20), upper: int=int(1e20), options: List[int]=[], suggestions: List[int]=[], additions: Any=[]):
    """check the value is a float data and is in range (lower, upper)

    Args:
        lower:
            the value should be greater than lower
        upper:
            the value should be less than upper
        options:
            the value should be in options
        additions:
            the values should skip all check, default values is null list, when there is only one addition value, you can set the value itself as additions ranther than a list.
    Returns:
        the value itself
    """
    if not isinstance(additions, List):
        additions = [additions]
    if lower is None:
        lower = -int(1e20)
    if upper is None:
        upper = int(1e20)

    def _(value: int, options=options, suggestions=suggestions, additions=additions):
        if value in additions:
            return value
        try:
            value = int(value)
        except Exception as e:
            raise e
        if value>=upper or value<=lower:
            raise ValueError(f"Value {value} is not in range ({lower}, {upper})")
        if options:
            if value not in options:
                raise ValueError(f"Value {value} is not in {options}")
        return value
    return _


def str_check(options: List[str]=[], suggestions: List[str]=[], additions: Any=[]):
    """check the value is a float data and is in range (lower, upper)

    Args:
        options:
            the value should be in options
        additions:
            the values should skip all check, default values is null list, when there is only one addition value, you can set the value itself as additions ranther than a list.
    Returns:
        the value itself
    """
    if not isinstance(additions, List):
        additions = [additions]
    def _(value: str, options=options, suggestions=suggestions, additions=additions):
        if value in additions:
            return value
        try:
            value = str(value)
        except Exception as e:
            raise e
        if options:
            if value not in options:
                raise ValueError(f"Value {value} is not in {options}")
        return value
    return _


def options(options: List[Any]=[], suggestions: List[Any]=[], additions: Any=[]):
    """check the value is in range options

    Args:
        options:
            the value should be in options
        additions:
            the values should skip all check, default values is null list, when there is only one addition value, you can set the value itself as additions ranther than a list.
    Returns:
        the value itself
    """
    if not isinstance(additions, List):
        additions = [additions]
    def _(value: int, options=options, suggestions=suggestions, additions=additions):
        if value in additions:
            return value
        if options:
            if value not in options:
                raise ValueError(f"Value {value} is not in {options}")
        return value
    return _


def suggestions(suggestions: List[Any]=[]):
    """provide suggestions
    Args:
        suggestions:
            the suggestion value
    Returns:
        the value itself
    """
    def _(value: Any, suggestions=suggestions):
        return value
    return _


def _extract_option_suggestion_additions(checker)->List[List[Any]]:
    """extract option and suggestion from checker parameters

    Args:
        checker: 
            the checker function

    Returns: 
        options and suggestions

    """
    parameters = inspect.signature(checker).parameters

    if 'options' in parameters:
        options = parameters['options'].default
    else:
        options = list()
    if "suggestions" in parameters:
        suggestions = parameters['suggestions'].default
    else:
        suggestions = list()
    if "additions" in parameters:
        additions = parameters['additions'].default
    else:
        additions = list()
    assert isinstance(options, list), options
    assert isinstance(suggestions, list), suggestions
    assert isinstance(additions, list), additions
    return [options, suggestions, additions]


def nest_converter(data_class: Any, options: List[str]=[]):
    """check the value is a float data and is in range (lower, upper)

    Args:
        options:
            the value should be in options
    Returns:
        the value itself
    """
    def _(value: dict, data_class=data_class):
        if isinstance(value, dict):
            return data_class(**value)
        else:
            return data_class(**asdict(value))
    return _


def IntField(value: Union[int, str, None], checker: Callable=int_check(), help: str="", type: Type=int)->int:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def BoolField(value: Union[None, bool, str], checker: Callable=options(options=[True, False]), help: str="", type: Type=bool)->bool:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def FloatField(value: Union[float, None, int, str], checker: Callable=float_check(), help: str="", type: Type=float)->float:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def NumberField(value: Union[float, None, int, str], checker: Callable=number_check(), help: str="", type: Type=float)->float:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def StrField(value: Union[str, None], checker: Callable=options(), help: str="", type: Type=str)->str:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def NameField(value: str, file, help: str="", type: Type=str)->_CountingAttr:
    return field(init=True, default=value, converter=None, type=type, metadata={"help": help, "path": os.path.abspath(file)})


def AnyField(value: Type[Dep], checker: Callable=options(), help: str="", type: Type=Type[Dep])->Type[Dep]:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def NestField(value: Type[Dep], converter: Callable, help: str="", type: Type=Type[Dep])->Type[Dep]:
    return field(default=value(), converter=converter(value), type=type, metadata={"help": help, "class": value})


def ListField(value: Union[List, None], checker: Callable=options(), help: str="", type: Type=List)->List:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def DictField(value: Union[Dict, None], checker: Callable=suggestions(), help: str="", type: Type=Dict)->Dict:
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    return field(init=True, default=value, converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})


def SubModules(value: Dict, checker: Callable=suggestions(), help: str="", type: Type=Dict)->Dict:
    def update_submodule_format(submodules):
        formated = {}
        for submodule, submodule_config in submodules.items():
            if isinstance(submodule_config, str):
                formated[submodule] = {"base": submodules[submodule]}
            else:
                assert isinstance(submodule_config, dict), submodule_config
                formated[submodule] = submodule_config
        return formated
    options, suggestions, additions = _extract_option_suggestion_additions(checker)
    options = [update_submodule_format(option) for option in options]
    suggestions = [update_submodule_format(suggestion) for suggestion in suggestions]
    return field(init=True, default=update_submodule_format(value), converter=checker, type=type, metadata={"help": help, "options": options, "suggestions": suggestions, "additions": additions})

@define
class BaseConfig:
    links = DictField({
    })
    @classmethod
    def from_dict(cls, config: Dict):
        new_config = {}
        for key in config:
            if key in {'name', 'config', 'links'}:
                new_config[key] = config[key]
            else:
                new_config['submods'] = new_config.get('submods', {})
                new_config['submods'][key] = config[key]
        return cls(**new_config)

    def to_dict(self)->Dict:
        result = asdict(self)
        submods = result.pop('submods', {})
        result.update(submods)
        return result

    def __str__(self):
        return json.dumps(self.to_dict())

# @define
# class ModuleConfig(BaseConfig):
#     name = NameField(value="basic", file=__file__, help="register module name")
#     @define
#     class Config:
#         input_map = DictField(value={}, help="name of model")
#         output_map = DictField(value={}, help="name of model")
#         dropout = FloatField(value=0.1, checker=float_check(-1, 100, options=[0.1,2 ,3]))
#         @define
#         class Nest:
#             nest_lr = FloatField(value=0.1, checker=float_check(-1, 100, options=[0.1,2 ,3]))
#         nest = NestField(value=Nest, converter=nest_converter)

#     config = NestField(value=Config, converter=nest_converter)
#     links = DictField({
#         "config.dropout,config.hidden_size@@lambda x,y: x**2": ["optimizer@adam.config.dropout"]
#     })
#     submods = SubModules({"optimizer@adam": "adam"}, checker=suggestions([{"optimizer": "adam", "optimizer": "sgd"}]))


# config = ModuleConfig.from_dict({"name": "nihao", "config": {"dropout": 0.1, "nest": {"nest_lr": 0.1}}, "optimizer@adam": {"base": "adam"}})
# config = ModuleConfig.from_dict({"name": "nihao", "config": {"dropout": 0.1, "nest": {"nest_lr": 0.1}}, "optimizer@adam": {"name": "adamw"}})


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
    def get_leaf_module(module_register, module_config_register, module_type_name: str, config: Dict, init: bool=False) -> Any:
        """get the module from module_register and module_config from module_config_register which name=module_name

        Args:
            module_register: register for module which has 'module_type_name'
            module_config_register: config register for config which has 'module_type_name'
            module_type_name: the module name which we want to get from register
            init: if True, we will init the module with config

        Returns: 
            module, module_config

        """
        if isinstance(config, str):
            name = config
            extend_config = {}
        else:
            assert isinstance(config, dict), "{} config must be name(str) or config(dict), but you provide {}".format(module_type_name, config)
            name = config.get('name', "") # must provide name
            extend_config = config
            if not name:
                raise KeyError(f'You must provide the {module_type_name} name("name")')

        module_class, module_config_class =  module_register.get(module_type_name, name), module_config_register.get(module_type_name, name)
        if (not module_class) or not (module_config_class):
            raise KeyError(f'The {module_type_name} name {config} is not registed.')
        module_config = module_config_class(extend_config)
        if not init:
            return module_class, module_config
        else:
            return module_class(module_config)

    @staticmethod
    def get_config_by_stage(stage:str, config:Dict)->Dict:
        """get the stage_config for special stage in provide config

        it means the config of this stage equals to config[stage]
        return config[config[stage]]

        Config Example:
            >>> config = {
            >>>     "train":{ //train、predict、online stage config
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
