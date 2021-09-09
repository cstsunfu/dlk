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


class BaseJsonConfigParser(object):
    """docstring for BaseJsonConfigParser"""
    def __init__(self, config_file_path: str, configure_base_path: str):
        """
        :config_file_path: the config file path
        :configure_base_path: the directory of __base__ configure
        """
        super(BaseJsonConfigParser, self).__init__()
        self.config_file_path = config_file_path
        self.model_config = self.load_hjson_file(self.config_file_path)
        self.config_name = self.model_config['name']
        self.config = self.model_config['config']
        self.configure_base_path = configure_base_path # must be rewrite
        

    def parser(self, update_config: dict={}) -> List[Dict]:
        """return a list of para dict s for all possible combinations
        :update_config: the update_config will update the returned config
        :returns: list[possible config dict]

        """
        raise NotImplementedError

    def load_hjson_file(self, file_name: str) -> Dict:
        """load hjson file by file_name

        :file_name: TODO
        :returns: TODO

        """
        json_file = hjson.load(open(file_name), object_pairs_hook=dict)
        return json_file

    def update_base(self, abstract_config: dict, kind_module: str="") -> dict:
        """TODO: Docstring for _update_base.

        :abstract_config: dict: TODO
        :kind_module: str: TODO
        :returns: TODO

        """
        module_base_config_name = abstract_config.get('__base__', '')
        if module_base_config_name == '':
            assert abstract_config.get('name', '') != '', print('You must provide a configure name for "{}" in root configure.'.format(kind_module))
            return abstract_config

        module_base_config = self.update_base(self.load_hjson_file(os.path.join(self.configure_base_path, kind_module+'s', module_base_config_name+'.hjson')), kind_module)
        assert abstract_config.get('name', '') == '', print('The module name must provide by root config in module "{}". But in child there is a name "{}" in configure.'.format(kind_module, abstract_config.get('name')))
        module_config = abstract_config.get('config', {})
        module_base_config['config'].update(module_config)
        
        return {'name': module_base_config['name'], 'config': module_base_config['config']}

    def flat_search(self, config: dict, kind_module: str='') -> List[dict]:
        """flat all the __search__ paras to list

        :config: dict: base config
        :kind_module: str: module kind name
        :returns: list of possible config

        """
        result = []
        module_search_para = config.get('__search__', None)
        if not module_search_para:
            result.append({
                'name': config['name'],
                'config': config['config']
            })
        else:
            search_para_list = self.get_cartesian_prod(module_search_para)
            for search_para in search_para_list:
                base_config = copy.deepcopy(config['config'])
                base_config.update(search_para)
                result.append({"name": config['name'], "config": base_config})

        return result

    def get_cartesian_prod(self, dict_of_list: Dict[str, List]) -> List[Dict]:
        """get catesian prod from named lists
        :dict_of_list: {'name1': [1,2,3], 'name2': [1,2,3]}
        :returns: [{'name1': 1, 'name2': 1}, {'name1': 1, 'name2': 2}, {'name1': 1, 'name2': 3}, ...]

        """
        if len(dict_of_list) == 0:
            return []
        cur_name, cur_paras  = dict_of_list.popitem()
        cur_para_search_list = []
        for para in cur_paras:
            cur_para_search_list.append({cur_name: para})
        if len(dict_of_list) == 0:
            return cur_para_search_list
        reserve_para_list = self.get_cartesian_prod(dict_of_list)
        all_config_list = []
        for cur_config in cur_para_search_list:
            for reserve_config in reserve_para_list:
                cur_config = copy.deepcopy(cur_config)
                cur_config.update(reserve_config)
                all_config_list.append(cur_config)

        return all_config_list

    def is_rep_config(self, list_of_dict: List[dict]) -> bool:
        """check is there a repeat config in list

        :list_of_dict: List[dict]: TODO
        :returns: TODO

        """
        list_of_str = [str(dic) for dic in list_of_dict]
        if len(list_of_dict) == len(set(list_of_str)):
            return False
        else:
            return True
        
class DataloaderJsonConfigParser(BaseJsonConfigParser):
    """docstring for DataloaderJsonConfigParser"""
    def __init__(self, config_file_path, configure_base_path='configures/dataloaders'):
        super(DataloaderJsonConfigParser, self).__init__(config_file_path=config_file_path, configure_base_path=configure_base_path)
        self.config = self.load_hjson_file(os.path.join(self.configure_base_path, self.config_name+'.hjson'))

    def parser(self, update_config: dict={}) -> List[Dict]:
        """return a list of para dict s for all possible combinations
        :update_config: the update_config will update the returned config
        :returns: list[possible config dict]

        """
        module_config = self.update_base(self.config)
        module_config.update(update_config)
        module_config_flat = self.flat_search(module_config)
        assert len(module_config_flat) == 1, print("The dataloader config must be unique.")
        return module_config_flat


class OptimizerJsonConfigParser(BaseJsonConfigParser):
    """docstring for ModelJsonConfigParser"""
    def __init__(self, config_file_path, configure_base_path='configures/optimizers'):
        super(OptimizerJsonConfigParser, self).__init__(config_file_path=config_file_path, configure_base_path=configure_base_path)
        self.config = self.load_hjson_file(os.path.join(self.configure_base_path, self.config_name+'.hjson'))

    def parser(self, update_config: dict={}) -> List[Dict]:
        """return a list of para dict s for all possible combinations
        :update_config: the update_config will update the returned config
        :returns: list[possible config dict]

        """
        module_config = self.update_base(self.config)
        module_config.update(update_config)
        module_config_flat = self.flat_search(module_config)
        assert len(module_config_flat) == 1, print("The optimizer config must be unique.")
        
        return module_config_flat


class TokenizerJsonConfigParser(BaseJsonConfigParser):
    """docstring for ModelJsonConfigParser"""
    def __init__(self, config_file_path, configure_base_path='configures/tokenizers'):
        super(TokenizerJsonConfigParser, self).__init__(config_file_path=config_file_path, configure_base_path=configure_base_path)
        self.config = self.load_hjson_file(os.path.join(self.configure_base_path, self.config_name+'.hjson'))

    def parser(self, update_config: dict={}) -> List[Dict]:
        """return a list of para dict s for all possible combinations
        :update_config: the update_config will update the returned config
        :returns: list[possible config dict]

        """
        module_config = self.update_base(self.config)
        module_config.update(update_config)
        module_config_flat = self.flat_search(module_config)
        assert len(module_config_flat) == 1, print("The tokenizer config must be unique.")
        
        return module_config_flat


class ModelJsonConfigParser(BaseJsonConfigParser):
    """docstring for ModelJsonConfigParser"""
    def __init__(self, config_file_path, configure_base_path='configures/modules'):
        super(ModelJsonConfigParser, self).__init__(config_file_path=config_file_path, configure_base_path=configure_base_path)

    def _update_config(self, config: dict, update_config: dict={}) ->Dict:
        """use update_config update the config

        :config: will updated config
        :returns: updated config

        """
        # new_config = update_config.get('config', {})
        for module in update_config:
            if module not in config:
                raise KeyError('The model config has not the module "{}"'.format(module))
            config[module]['config'].update(update_config[module].get('config', {}))
            config[module]['__search__'] = update_config[module].get('__search__', {})

        return config

    def parser(self, update_config: dict={}) -> List[Dict]:
        """ return a list of para dict s for all possible combinations
        :update_config: the update_config will update the returned config
        :returns: list[possible config dict]
        """

        modules_config = self._parser(self.config, self.update_base)
        modules_config = self._update_config(modules_config, update_config)
        all_module_config_in_one = self._parser(modules_config, self.flat_search)
        all_module_config_list = self.get_cartesian_prod(all_module_config_in_one)
        return_list = []
        for config in all_module_config_list:
            return_list.append({
                'name': self.config_name,
                'config': config
                })
        check = self.is_rep_config(return_list)
        if check:
            print('Please check your paras carefully, there are repeat configs!!')
            for config in return_list:
                print(config)
            raise ValueError('REPEAT CONFIG')
        return return_list

    def _parser(self, config: dict, map_fun: Callable) -> Dict:
        """use the map_fun to process all the modules
        :returns: depend on 

        """
        modules_config = {}
        for kind_module in config:
            modules_config[kind_module] = map_fun(config[kind_module], kind_module)
        return modules_config



class ProcessorJsonConfigParser(BaseJsonConfigParser):
    """docstring for ProcessorJsonConfigParser"""
    def __init__(self, config_file_path, configure_base_path='configures/modules'):
        super(ProcessorJsonConfigParser, self).__init__(config_file_path=config_file_path, configure_base_path=configure_base_path)

    def _update_config(self, config: dict, update_config: dict={}) ->Dict:
        """use update_config update the config

        :config: will updated config
        :returns: updated config

        """
        # new_config = update_config.get('config', {})
        for module in update_config:
            if module not in config:
                raise KeyError('The processor config has not the module "{}"'.format(module))
            config[module]['config'].update(update_config[module].get('config', {}))

        return config

    def parser(self, update_config: dict={}) -> List[Dict]:
        """ return a list of para dict s for all possible combinations
        :update_config: the update_config will update the returned config
        :returns: list[possible config dict]
        """

        modules_config = self._parser(self.config, self.update_base)
        modules_config = self._update_config(modules_config, update_config)
        all_module_config_in_one = self._parser(modules_config, self.flat_search)
        all_module_config_list = self.get_cartesian_prod(all_module_config_in_one)
        return_list = []
        for config in all_module_config_list:
            return_list.append({
                'name': self.config_name,
                'config': config
                })
        check = self.is_rep_config(return_list)
        if check:
            print('Please check your paras carefully, there are repeat configs!!')
            for config in return_list:
                print(config)
            raise ValueError('REPEAT CONFIG')
        if len(all_module_config_list) != 1:
            raise ValueError('Currently the project is not support search para in processor.')

        return return_list

    def _parser(self, config: dict, map_fun: Callable) -> Dict:
        """use the map_fun to process all the modules
        :returns: depend on 

        """
        modules_config = {}
        for kind_module in config:
            modules_config[kind_module] = map_fun(config[kind_module], kind_module)
        return modules_config
