import hjson
import os
import copy
from typing import Callable, List, Dict, Union
from dlkit.utils.register import Register
from dlkit.utils.config import ConfigTool
import json

config_parser_register = Register("Config parser register")


class BaseConfigParser(object):
    """docstring for BaseConfigParser

        input: config_file, config_base_dir
        config_name = config_file._name
        search = config_file._search
        link = config_file._link
        base = self.config_file._base
        if base:
            base_config = parser(base)
    """
    def __init__(self, config_file: Union[str, Dict, List], config_base_dir: str=""):
        super(BaseConfigParser, self).__init__()
        if isinstance(config_file, str):
            try:
                self.config_file = self.load_hjson_file(os.path.join(config_base_dir, config_file+'.hjson'))
            except:
                print(config_base_dir, config_file)
                raise KeyError
        elif isinstance(config_file, Dict):
            self.config_file = config_file
        else:
            raise KeyError('The config file must be str or dict. You provide {}.'.format(config_file))

        self.config_name = self.config_file.pop('_name', "")
        self.search = self.config_file.pop('_search', {})
        self.link = self.config_file.pop('_link', {})

        base = self.config_file.pop('_base', "")
        self.base_config = {}
        if base:
            self.base_config = self.get_base_config(base)
        if self.base_config and self.config_name:
            raise PermissionError("You should put the _name to the leaf config.")
        self.modules = self.config_file
            

    @classmethod
    def get_base_config(cls, config_file)->Dict:
        """TODO: Docstring for get_base_config.

        :arg1: TODO
        :returns: TODO

        """
        base_config = cls(config_file).parser()
        if len(base_config)>1:
            raise PermissionError("The base config don't support _search now.")
        if base_config:
            return base_config[0]
        return {}

    @classmethod
    def config_link_para(cls, link: Dict[str, Union[str, List[str]]]={}, config: Dict={}):
        """link the config[to] = config[source]
        :link: {source1:to1, source2:[to2, to3]}
        :returns:
        """
        def make_link(source: str, to: str):
            """copy the 'source' config to 'to'
            """
            source_config = config
            to_config = config
            source_list = source.split('.')
            to_list = to.split('.')
            for s, t in zip(source_list[:-1], to_list[:-1]):
                if isinstance(source_config, list):
                    ss = s
                    if s[0] == '-':
                        ss = s[1:]
                    assert str.isdigit(ss)
                    s = int(s)
                if isinstance(to_config, list):
                    tt = t
                    if t[0] == '-':
                        tt = t[1:]
                    assert str.isdigit(tt), print("for list index must be int")
                    t = int(t)
                source_config = source_config[s]
                to_config = to_config[t]
            to_config[to_list[-1]] = source_config[source_list[-1]]

        if not link:
            return
        for (source, to) in link.items():
            if isinstance(to, List):
                for sub_to in to:
                    make_link(source, sub_to)
            else:
                make_link(source, to)

    def parser_with_check(self)->List[Dict]:
        """parser and check the result, only used for __main__ processor
        :returns: TODO
        """
        configs = self.parser()
        self.check_config(configs)
        return configs

    def parser(self) -> List[Dict]:
        """ return a list of para dicts for all possible combinations
        :returns: list[possible config dict]
        """

        # parser submodules get submodules config
        modules_config = self.map_to_submodule(self.modules, self.get_kind_module_base_config)

        # expand all submodules to combine a set of module configs
        possible_config_list = self.get_named_list_cartesian_prod(modules_config)

        # using base_config to update all possible_config
        if possible_config_list:
            possible_config_list = [ConfigTool.do_update_config(self.base_config, possible_config) for possible_config in possible_config_list]
        else:
            possible_config_list = [self.base_config]

        # flat all search paras
        possible_config_list_list = [self.flat_search(self.search, possible_config) for possible_config in possible_config_list]

        all_possible_config_list = []
        for possible_config_list in possible_config_list_list:
            all_possible_config_list.extend(possible_config_list)

        # link paras
        for possible_config in all_possible_config_list:
            self.config_link_para(self.link, possible_config)

        return_list = []
        for possible_config in all_possible_config_list:
            config = copy.deepcopy(possible_config)
            if self.config_name:
                config['_name'] = self.config_name
            return_list.append(config)

        if self.is_rep_config(return_list):
            for config in return_list:
                print(config)
            raise ValueError('REPEAT CONFIG')
        return return_list

    def get_kind_module_base_config(self, abstract_config: Union[dict, str], kind_module: str="") -> List[dict]:
        """get the whole config of 'kind_module' by given abstract_config

        :abstract_config: dict: will expanded config
        :returns: parserd config (whole config) of abstract_config

        """
        return config_parser_register.get(kind_module)(abstract_config).parser()

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

    # def flat_search(self, config: dict) -> List[dict]:
    @classmethod
    def flat_search(cls, search, config: dict) -> List[dict]:
        """flat all the _search paras to list
        support recursive parser _search now, this means you can add _search/_link/_base paras in _search paras

        :config: dict: base config
        :returns: list of possible config

        """
        result = []
        module_search_para = search
        if not module_search_para:
            result.append(config)
        else:
            search_para_list = cls.get_named_list_cartesian_prod(module_search_para)
            for search_para in search_para_list:
                base_config = copy.deepcopy(config)
                base_config.update(search_para)
                extend_config = cls(base_config).parser()
                result.extend(extend_config)
                # result.append(base_config)

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

    @staticmethod
    def check_config(configs: Union[Dict, List[Dict]]) -> None:
        """check all config is right.
        1. check all "*@*" is replaced to correct value.
        :configs: one config or a list of configs
        """
        def _check(config):
            """TODO: Docstring for _check.

            :config: TODO
            :returns: TODO

            """
            for key in config:
                if isinstance(config[key], dict):
                    _check(config[key])
                if config[key] == '*@*':
                    raise ValueError(f'In Config: \n {json.dumps(config, indent=4)}\n The must be provided key "{key}" marked with "*@*" is not provided.')

        if isinstance(configs, list):
            for config in configs:
                _check(config)
        else:
            _check(configs)

    @staticmethod
    def get_named_list_cartesian_prod(dict_of_list: Dict[str, List]={}) -> List[Dict]:
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
        reserve_para_list = BaseConfigParser.get_named_list_cartesian_prod(dict_of_list)
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
        :returns: is there the list of dict repeat

        """
        # using json.dumps + sort_keys to guarantee the same dict to the same string represatation
        list_of_str = [json.dumps(dic, sort_keys=True) for dic in list_of_dict]
        if len(list_of_dict) == len(set(list_of_str)):
            return False
        else:
            return True


@config_parser_register('task')
class TaskConfigParser(BaseConfigParser):
    """docstring for TaskConfigParser"""
    def __init__(self, config_file):
        super(TaskConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/tasks/')


@config_parser_register('model')
class ModelConfigParser(BaseConfigParser):
    """docstring for ModelConfigParser"""
    def __init__(self, config_file):
        super(ModelConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/models/')
        

@config_parser_register('optimizer')
class OptimizerConfigParser(BaseConfigParser):
    """docstring for OptimizerConfigParser"""
    def __init__(self, config_file):
        super(OptimizerConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/optimizers/')


@config_parser_register('loss')
class LossConfigParser(BaseConfigParser):
    """docstring for LossConfigParser"""
    def __init__(self, config_file):
        super(LossConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/losses/')

@config_parser_register('config')
class ConfigConfigParser(BaseConfigParser):
    """docstring for ConfigConfigParser"""
    def __init__(self, config_file):
        super(ConfigConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/configs/')
        if self.base_config:
            raise AttributeError('The paras config do not support _base.')
        if self.link:
            raise AttributeError('The paras config do not support _link.')
        if self.config_name:
            raise AttributeError('The paras config do not support _name.')

    def parser(self, update_config: Dict={}):
        """TODO: Docstring for parser.

        :update_config: Dict: TODO
        :returns: TODO
        """
        config_list = self.flat_search(self.search, self.modules)
        return config_list


@config_parser_register('encoder')
class EncoderConfigParser(BaseConfigParser):
    """docstring for EncoderConfigParser"""
    def __init__(self, config_file):
        super(EncoderConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/encoders/')


@config_parser_register('decoder')
class DecoderConfigParser(BaseConfigParser):
    """docstring for DecoderConfigParser"""
    def __init__(self, config_file):
        super(DecoderConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/decoders/')


@config_parser_register('embedding')
class EmbeddingConfigParser(BaseConfigParser):
    """docstring for EmbeddingConfigParser"""
    def __init__(self, config_file):
        super(EmbeddingConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/embeddings/')


@config_parser_register('module')
class ModuleConfigParser(BaseConfigParser):
    """docstring for ModuleConfigParser"""
    def __init__(self, config_file):
        super(ModuleConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/modules/')

@config_parser_register('processor')
class ProcessorConfigParser(BaseConfigParser):
    """docstring for ProcessorConfigParser"""
    def __init__(self, config_file):
        super(ProcessorConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/processors/')


@config_parser_register('subprocessor')
class SubProcessorConfigParser(BaseConfigParser):
    """docstring for SubProcessorConfigParser"""
    def __init__(self, config_file):
        super(SubProcessorConfigParser, self).__init__(config_file, config_base_dir='dlkit/configures/subprocessors/')
