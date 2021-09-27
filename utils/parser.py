import hjson
import os
import copy
from typing import Callable, List, Dict, Union

CONFIG_PARSER_REGISTRY = {}


def config_parser_register(name: Union[str, List[str]] = "") -> Callable:
    """
    register config parsers
    """

    def decorator(model):
        def register(sub_name: str=""):
            """TODO: Docstring for register.

            :sub_name: TODO
            :returns: TODO

            """
            if sub_name.strip() == "":
                raise ValueError('You must set a name for {}'.format(model.__name__))

            if sub_name in CONFIG_PARSER_REGISTRY:
                raise ValueError('The config parser name {} is already registed.'.format(name))
            CONFIG_PARSER_REGISTRY[sub_name] = model

        if isinstance(name, List):
            for sub_name in name:
                register(sub_name)
        else:
            register(name)
        return model
    return decorator


class BaseConfigParser(object):
    """docstring for BaseConfigParser"""
    def __init__(self, config_file: Union[str, Dict, List], config_base_dir: str=""):
        super(BaseConfigParser, self).__init__()
        if isinstance(config_file, str):
            self.config_file = self.load_hjson_file(os.path.join(config_base_dir, config_file+'.hjson'))
        elif isinstance(config_file, Dict):
            self.config_file = config_file
        else:
            raise KeyError('The config file must be str or dict. You provide {}.'.format(config_file))

        self.config_name = self.config_file.pop('__name', "")
        self.search = self.config_file.pop('__search', {})
        self.link = self.config_file.pop('__link', {})

        base = self.config_file.pop('__base', "")
        self.base_config = {}
        if base:
            self.base_config = self.get_base_config(base)
        if self.base_config and self.config_name:
            raise PermissionError("You should put the __name to the leaf config.")
        self.modules = self.config_file
            

    @classmethod
    def get_base_config(cls, config_file, update_config: dict={})->Dict:
        """TODO: Docstring for get_base_config.

        :arg1: TODO
        :returns: TODO

        """
        base_config = cls(config_file).parser(update_config)
        if len(base_config)>1:
            raise PermissionError("The base config don't support __search now.")
        if base_config:
            return base_config[0]
        return {}

    def config_link_para(self, link: Dict={}, config: Dict={}):
        """link the self.config[to] = self.config[source]
        :link: {source1:to1, ...}
        :returns:
        """
        if not link:
            return
        for (source, to) in link.items():
            source_config = config
            to_config = config
            source_list = source.split('.')
            to_list = to.split('.')
            for s, t in zip(source_list[:-1], to_list[:-1]):
                source_config = source_config[s]
                to_config = to_config[t]
            to_config[to_list[-1]] = source_config[source_list[-1]]

    def parser(self, update_config: dict={}) -> Union[List[Dict], List]:
        """ return a list of para dicts for all possible combinations
        :returns: list[possible config dict]
        :returns: list of possible config dict list(group)
        """

        modules_config = self.map_to_submodule(self.modules, self.get_kind_module_base_config)
        possible_config_list = self.get_named_list_cartesian_prod(modules_config)
        if possible_config_list:
            possible_config_list = [self.do_update_config(self.base_config, possible_config) for possible_config in possible_config_list]
        else:
            possible_config_list = [self.base_config]

        possible_config_list = [self.do_update_config(possible_config, update_config) for possible_config in possible_config_list]


        possible_config_list_list = [self.flat_search(possible_config) for possible_config in possible_config_list]

        all_possible_config_list = []
        for possible_config_list in possible_config_list_list:
            # all_possible_config_list.extend(self.get_named_list_cartesian_prod(possible_config_list))
            all_possible_config_list.extend(possible_config_list)
        for possible_config in all_possible_config_list:
            self.config_link_para(self.link, possible_config)

        return_list = []
        for possible_config in all_possible_config_list:
            config = copy.deepcopy(possible_config)
            if self.config_name:
                config['__name'] = self.config_name
            return_list.append(config)

        if self.is_rep_config(return_list):
            for config in return_list:
                print(config)
            raise ValueError('REPEAT CONFIG')
        return return_list


    def do_update_config(self, config: dict, update_config: dict={}) ->Dict:
        """use update_config update the config

        :config: will updated config
        :returns: updated config

        """
        def _inplace_update_dict(_base, _new):
            """TODO: Docstring for rec_update_dict.
            :returns: TODO

            """
            for item in _new:
                if (item not in _base) or ((not isinstance(_new[item], Dict) and (not isinstance(_base[item], Dict)))):
                # if item not in _base, or they all are not Dict
                    _base[item] = _new[item]
                elif isinstance(_base[item], Dict) and isinstance(_new[item], Dict):
                    _inplace_update_dict(_base[item], _new[item])
                else:
                    raise AttributeError("The base config and update config is not match. base: {}, new: {}. ".format(_base, _new))
                

        # new_config = update_config.get('config', {})
        config = copy.deepcopy(config)
        _inplace_update_dict(config, update_config)
        return config

    def get_kind_module_base_config(self, abstract_config: Union[dict, str], kind_module: str="") -> List[dict]:
        """TODO: Docstring for _get_kind_module_base_config.

        :abstract_config: dict: TODO
        :returns: TODO
        :returns: 

        """
        return CONFIG_PARSER_REGISTRY[kind_module](abstract_config).parser()

    def map_to_submodule(self, config: dict, map_fun: Callable) -> Dict:
        """use the map_fun to process all the modules
        :returns: depend on 

        """
        modules_config = {}
        for kind_module in config:
            modules_config[kind_module] = map_fun(config[kind_module], kind_module)
        return modules_config

    def load_hjson_file(self, file_name: str) -> Dict:
        """load hjson file by file_name

        :file_name: TODO
        :returns: TODO

        """
        json_file = hjson.load(open(file_name), object_pairs_hook=dict)
        return json_file

    def flat_search(self, config: dict, _: str='', extend_base=False) -> List[dict]:
        """flat all the __search paras to list

        :config: dict: base config
        :kind_module: str: module kind name
        :returns: list of possible config

        """
        result = []
        module_search_para = self.search
        if not module_search_para:
            result.append(config)
        else:
            search_para_list = self.get_named_list_cartesian_prod(module_search_para)
            # TODO: support search __base when extend_base is True
            for search_para in search_para_list:
                base_config = copy.deepcopy(config)
                base_config.update(search_para)
                result.append(base_config)

        return result

    def get_cartesian_prod(self, list_of_list_of_dict: List[List[Dict]]) -> List[List[Dict]]:
        """get catesian prod from two lists
        :list_of_list_of_dict: [[config_a1, config_a2], [config_b1, config_b2]]
        :returns: [[config_a1, config_b1],
                   [config_a1, config_b2],
                   [config_a2, config_b1],
                   [config_a2, config_b2]
                  ]
        """
        if len(list_of_list_of_dict) <= 1:
            return [copy.deepcopy(dic) for dic in list_of_list_of_dict]
        cur_result = list_of_list_of_dict[0]
        reserve_result = self.get_cartesian_prod(list_of_list_of_dict[1:])
        result = []
        for cur_config in cur_result:
            for reserve in reserve_result:
                result.append([copy.deepcopy(cur_config)]+copy.deepcopy(reserve))
        return result

    def get_named_list_cartesian_prod(self, dict_of_list: Dict[str, List]={}) -> List[Dict]:
        """get catesian prod from named lists
        :dict_of_list: {'name1': [1,2,3], 'name2': [1,2,3]}
        :returns: [{'name1': 1, 'name2': 1}, {'name1': 1, 'name2': 2}, {'name1': 1, 'name2': 3}, ...]

        """
        if len(dict_of_list) == 0:
            return []
        dict_of_list = copy.deepcopy(dict_of_list)
        cur_name, cur_paras  = dict_of_list.popitem()
        cur_para_search_list = []
        for para in cur_paras:
            cur_para_search_list.append({cur_name: para})
        if len(dict_of_list) == 0:
            return cur_para_search_list
        reserve_para_list = self.get_named_list_cartesian_prod(dict_of_list)
        all_config_list = []
        for cur_config in cur_para_search_list:
            for reserve_config in reserve_para_list:
                _cur_config = copy.deepcopy(cur_config)
                _cur_config.update(copy.deepcopy(reserve_config))
                all_config_list.append(_cur_config)
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


@config_parser_register('system')
class SystemConfigParser(BaseConfigParser):
    """docstring for SystemConfigParser"""
    def __init__(self, config_file):
        if isinstance(config_file, dict) and '__search' in config_file:
            raise KeyError('The system(task) config is not support __search now.')
        super(SystemConfigParser, self).__init__(config_file, config_base_dir='configures/tasks/')


@config_parser_register(['model', 'model_student', "model_teacher"] + ['model_'+str(i) for i in range(64)])
class ModelConfigParser(BaseConfigParser):
    """docstring for ModelConfigParser"""
    def __init__(self, config_file):
        super(ModelConfigParser, self).__init__(config_file, config_base_dir='configures/models/')
        

@config_parser_register('multi_models')
class MultiModelConfigParser(BaseConfigParser):
    """docstring for MultiModelConfigParser"""
    def __init__(self, config_file):
        super(MultiModelConfigParser, self).__init__(config_file, config_base_dir='configures/multi_models/')


@config_parser_register('dataloader')
class DataloaderConfigParser(BaseConfigParser):
    """docstring for DataloaderConfigParser"""
    def __init__(self, config_file):
        super(DataloaderConfigParser, self).__init__(config_file, config_base_dir='configures/dataloaders/')
        

@config_parser_register('optimizer')
class OptimizerConfigParser(BaseConfigParser):
    """docstring for OptimizerConfigParser"""
    def __init__(self, config_file):
        super(OptimizerConfigParser, self).__init__(config_file, config_base_dir='configures/optimizers/')


@config_parser_register('multi_optimizers')
class MultiOptimizerConfigParser(BaseConfigParser):
    """docstring for MultiOptimizerConfigParser"""
    def __init__(self, config_file):
        super(MultiOptimizerConfigParser, self).__init__(config_file, config_base_dir='configures/multi_optimizers/')


@config_parser_register('config')
class ConfigConfigParser(BaseConfigParser):
    """docstring for ConfigConfigParser"""
    def __init__(self, config_file):
        super(ConfigConfigParser, self).__init__(config_file, config_base_dir='configures/configs/')
        if self.base_config:
            raise AttributeError('The paras config do not support __base.')
        if self.link:
            raise AttributeError('The paras config do not support __link.')
        if self.config_name:
            raise AttributeError('The paras config do not support __name.')

    def parser(self, update_config: Dict={}):
        """TODO: Docstring for parser.

        :update_config: Dict: TODO
        :returns: TODO
        """
        config_list = self.flat_search(self.modules)
        return config_list


@config_parser_register('encoder')
class EncoderConfigParser(BaseConfigParser):
    """docstring for EncoderConfigParser"""
    def __init__(self, config_file):
        super(EncoderConfigParser, self).__init__(config_file, config_base_dir='configures/modules/encoders/')


@config_parser_register('decoder')
class DecoderConfigParser(BaseConfigParser):
    """docstring for DecoderConfigParser"""
    def __init__(self, config_file):
        super(DecoderConfigParser, self).__init__(config_file, config_base_dir='configures/modules/decoders/')


@config_parser_register('embedding')
class EmbeddingConfigParser(BaseConfigParser):
    """docstring for EmbeddingConfigParser"""
    def __init__(self, config_file):
        super(EmbeddingConfigParser, self).__init__(config_file, config_base_dir='configures/modules/embeddings/')
