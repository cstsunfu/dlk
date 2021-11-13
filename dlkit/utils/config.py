from typing import Any, Dict, Union, Callable, List, Tuple
import json
import copy
import os
from dlkit.utils.logger import get_logger
# import hjson


class GetConfigByStageMixin(object):
    """docstring for GetConfigByStageMixin"""

    def get_config(self, stage:str, config:Dict)->Dict:
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
            self.do_update_config(stage_config, stage_config[1])
        return stage_config


class Config(object):
    """docstring for Config"""
    def __init__(self, **kwargs):
        super(Config, self).__init__()
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                get_logger().error(f"Can't set {key} with value {value} for {self}")
                raise err
        
    def _get_leaf_module(self, module_register: Dict, module_config_register: Dict, module_name: str, config: Dict) -> Tuple[Any, 'Config']:
        """get sub module and config from register.
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
            for key in config:
                if key not in ['_name', 'config']:
                    raise KeyError('You can only provide the {} name("name") and config("config")'.format(module_name))
            name = config.get('_name', "") # must provide _name_
            extend_config = config.get('config', {})
            if not name:
                raise KeyError('You must provide the {} name("name")'.format(module_name))

        module, module_config =  module_register.get(name), module_config_register.get(name)
        if (not module) or not (module_config):
            raise KeyError('The {} name {} is not registed.'.format(module_name, config))
        module_config = Config.update(module_config, extend_config)
        return module, module_config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "Config":
        """
        Args:
            config_dict (:obj:`Dict[str, Any]`):
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`Config`: The configuration object instantiated from those parameters.
        """
        for key, value in kwargs.items():
            config_dict[key] = value
        config = cls(**config_dict)

        return config

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``Config()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=4, sort_keys=True, ensure_ascii=False) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``Config()`` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @staticmethod
    def _inplace_update_dict(_base, _new):
        """TODO: Docstring for _inplace_update_dict.
        :returns: TODO

        """
        for item in _new:
            if (item not in _base) or ((not isinstance(_new[item], Dict) and (not isinstance(_base[item], Dict)))):
            # if item not in _base, or they all are not Dict
                _base[item] = _new[item]
            elif isinstance(_base[item], Dict) and isinstance(_new[item], Dict):
                Config._inplace_update_dict(_base[item], _new[item])
            else:
                raise AttributeError("The base config and update config is not match. base: {}, new: {}. ".format(_base, _new))

    def do_update_config(self, config: dict, update_config: dict={}) ->Dict:
        """use update_config update the config

        :config: will updated config
        :returns: updated config

        """
        config = copy.deepcopy(config)
        self._inplace_update_dict(config, update_config)
        return config

    @classmethod
    def update(cls, base: "Config", config_dict: Dict[str, Any])->'Config':
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that shall be updated for this class.
        """
        # new_config = update_config.get('config', {})
        config = base.to_dict()
        cls._inplace_update_dict(config, config_dict)
        return cls(**config)
