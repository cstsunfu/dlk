from typing import Any, Dict, Union, Callable, List, Tuple
import json
import copy
import os
from utils.logger import get_logger
import hjson


class Config(object):
    """docstring for Config"""
    def __init__(self, **kwargs):
        super(Config, self).__init__()
        self.name = "base_config"
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                get_logger().error(f"Can't set {key} with value {value} for {self}")
                raise err
        
    def _get_sub_module(self, module_register: Dict, module_config_register: Dict, module_name: str, config: Dict) -> Tuple[Any, 'Config']:
        """get sub module and config from register.

        :module_register: TODO
        :module_config_register: TODO
        :module_name: TODO
        :config: Dict: TODO
        :returns: TODO

        """
        if isinstance(config, str):
            name = config
            extend_config = {}
        else:
            assert isinstance(config, dict), "{} config must be name(str) or config(dict), but you provide {}".format(module_name, config)
            for key in config:
                if key not in ['name', 'config']:
                    raise KeyError('You can only provide the {} name("name") and embedding config("config")'.format(module_name))
            name = config.get('name')
            extend_config = config.get('config', {})
            if not name:
                raise KeyError('You must provide the {} name("name")'.format(module_name))

        module, module_config =  module_register.get(name), module_config_register.get(name)
        if (not module) or not (module_config):
            raise KeyError('The {} name {} is not registed.'.format(module_name, config))
        module_config.update(extend_config)
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

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                get_logger().error(f"Can't set {key} with value {value} for {self}")
                raise err
