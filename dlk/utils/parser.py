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

import hjson
import os
import copy
from typing import Callable, List, Dict, Union
from dlk.utils import register
from dlk.utils.register import Register
from dlk.utils.config import ConfigTool
from dlk.utils.logger import Logger
from dlk.utils.get_root import get_root
from dlk.utils.register import Register
from dlk.core import (
    embedding_config_register,
    embedding_register,
    callback_config_register,
    callback_register,
    decoder_config_register,
    decoder_register,
    encoder_config_register,
    encoder_register,
    imodel_config_register,
    imodel_register,
    initmethod_config_register,
    initmethod_register,
    model_config_register,
    model_register,
    scheduler_config_register,
    loss_config_register,
    loss_register,
    module_config_register,
    module_register,
    optimizer_config_register,
    optimizer_register,
    scheduler_register
)
from dlk.data import (
    datamodule_config_register,
    datamodule_register,
    postprocessor_config_register,
    postprocessor_register,
    processor_config_register,
    processor_register,
    subprocessor_config_register,
    subprocessor_register
)
from dlk.managers import (
    manager_config_register, 
    manager_register
)
import json

logger = Logger.get_logger()

config_parser_register = Register("Config parser register")


class LinkUnionTool(object):
    """Assisting tool for parsering the "_link" of config. All the function named the top level has high priority than low level

    This class is mostly for resolve the confilicts of the low and high level register links.
    """
    def __init__(self):
        self.link_union = {}

    def find(self, key: str):
        """find the root of the key

        Args:
            key: a token

        Returns: 
            the root of the key

        """

        if key not in self.link_union:
            return None

        if self.link_union[key] != key:
            return self.find(self.link_union[key])
        return self.link_union[key]

    def low_level_union(self, link_from: str, link_to: str):
        """union the low level link_from->link_to pair

        On the basis of the high-level links, this function is used to regist low-level link
        If `link-from` and `link-to` were all not appeared at before, they will be directly registed.
        If only one of the `link-from` and `link-to` appeared, the value of the `link-from` and `link-to` will be overwritten by the corresponding value of the upper level,
        If both `link-from` and `link-to` appeared at before, and if they linked the same value, we will do nothing, otherwise `RAISE AN ERROR`

        Args:
            link_from: the link-from key
            link_to: the link-to key

        Returns: 
            None

        """
        if self.find(link_from) and self.find(link_to):  # all has been linked
            if self.find(link_from)!= self.find(link_to):
                raise PermissionError(f"The  {link_from} and {link_to} has been linked to different values, but now you want to link them together.")
            elif self.find(link_from) == link_to:
                logger.warning(f"High level config has the link '{link_to} -> {link_from}', and the low level reversed link '{link_from} -> {link_to}' is been ignored.")
            else:
                return
        elif self.find(link_to): # only link_to has been linked
            logger.warning(f"Parameter '{link_to}' has been linked in high level config, the link '{link_from} -> {link_to}' is invalid, and the real link is been reversed as '{link_to} -> {link_from}'.")
            self.link_union[link_from] = self.find(link_to)
        elif self.find(link_from): # only link_from has been linked
            self.link_union[link_to] = self.find(link_from)
        else:
            self.link_union[link_from] = link_from
            self.link_union[link_to] = link_from

    def top_level_union(self, link_from: str, link_to: str):
        """union the top level link_from->link_to pair

        Register the 'link'(`link-from` -> `link-to`) in the same(top) level config should be merged using `top_level_union`
        Parameters are not allowed to be assigned repeatedly (the same parameter cannot appear more than once in the `link-to` position, otherwise it will cause ambiguity.)

        Args:
            link_from: the link-from key
            link_to: the link-to key

        Returns: 
            None

        """
        if link_from not in self.link_union:
            self.link_union[link_from] = link_from

        assert link_to not in self.link_union, f"{link_to} is repeated assignment"
        self.link_union[link_to] = self.find(link_from)

    def register_top_links(self, links: Dict):
        """register the top level links, top level means the link_to level config

        Args:
            links: {"from": ["tolist"], "from2": "to2"}

        Returns: 
            self

        """
        for source, target in links.items():
            if isinstance(target, list):
                [self.top_level_union(source, t) for t in target]
            else:
                self.top_level_union(source, target)
        return self

    def register_low_links(self, links: Dict):
        """register the low level links, low level means the base(parant) level config

        Args:
            links: {"link-from": ["list of link-to"], "link-from2": "link-to2"}

        Returns: 
            self

        """
        for source, target in links.items():
            if isinstance(target, list):
                [self.low_level_union(source, t) for t in target]
            else:
                self.low_level_union(source, target)
        return self

    def get_links(self):
        """get the registed links

        Returns: 
            all registed and validation links

        """
        links = {}
        for key in self.link_union:
            root = self.find(key)
            if root == key:
                continue
            if root not in links:
                links[root] = []
            links[root].append(key)
        return links


class BaseConfigParser(object):
    """BaseConfigParser
    The config parser order is: inherit -> search -> link

    If some config is marked to "*@*", this means the para has not default value, you must coverd it(like 'label_nums', etc.).

    """
    def __init__(self, config_file: Union[str, Dict, List], config_base_dir: str="", register: Register=None):
        super(BaseConfigParser, self).__init__()
        if isinstance(config_file, str):
            if config_file == '*@*':
                self.config_file = "*@*"
                return
            try:
                if os.path.isfile(os.path.join(get_root(), config_base_dir, config_file+'.hjson')):
                    self.config_file = self.load_hjson_file(os.path.join(get_root(), config_base_dir, config_file+'.hjson'))
                else:
                    self.config_file = register.get(config_file).default_config

            except Exception as e:
                logger.error(f"There is an error occur when loading {os.path.join(get_root(), config_base_dir, config_file)}")
                raise KeyError(e)
        elif isinstance(config_file, Dict):
            self.config_file = config_file
        else:
            raise KeyError('The config file must be str or dict. You provide {}.'.format(config_file))

        self.config_name = self.config_file.pop('_name', "")
        self.search = self.config_file.pop('_search', {})
        base = self.config_file.pop('_base', "")
        self.base_config = {}
        if base:
            self.base_config = self.get_base_config(base)
        if "_focus" in self.config_file:
            self.base_config['_focus'] = self.config_file.pop('_focus')

        # merge base and current config _link
        link_union = LinkUnionTool()
        link_union.register_top_links(self.config_file.pop('_link', {}))
        link_union.register_low_links(self.base_config.pop('_link', {}))
        self.config_file['_link'] = link_union.get_links()

        if self.base_config and self.config_name:
            raise PermissionError("You should put the _name to the leaf config.")
        self.modules = self.config_file


    @classmethod
    def get_base_config(cls, config_name: str)->Dict:
        """get the base config use the config_name

        Args:
            config_name: the config name

        Returns: 
            config of the config_name
        """
        base_config = cls(config_name).parser(parser_link=False)
        if len(base_config)>1:
            raise PermissionError("The base config don't support _search now.")
        if base_config:
            return base_config[0]
        return {}

    @staticmethod
    def config_link_para(link: Dict[str, Union[str, List[str]]]=None, config: Dict=None):
        """inplace link the config[to] = config[source]

        Args:
            link: {link-from:link-to-1, link-from:[link-to-2, link-to-3]}
            config: will linked base config

        Returns: 
            None

        """
        if not link:
            link = {}
        if not config:
            config = {}
        def make_link(source: str, to: str):
            """copy the 'source' config to 'to'
            """
            try:
                source_config = config
                to_config = config
                source_list = source.split('.')
                to_list = to.split('.')
                for s in source_list[:-1]:
                    if isinstance(source_config, list):
                        assert (s[0]=='-' and str.isdigit(s[1:])) or str.isdigit(s), "list index must be int"
                        s = int(s)
                    source_config = source_config[s]
                for t in to_list[:-1]:
                    if isinstance(to_config, list):
                        assert (t[0]=='-' and str.isdigit(t[1:])) or str.isdigit(t), "list index must be int"
                        t = int(t)
                    to_config = to_config[t]
                if isinstance(to_config, list):
                    assert (to_list[-1][0]=='-' and str.isdigit(to_list[-1][1:])) or str.isdigit(to_list[-1]), "list index must be int"
                    to_list[-1] = int(to_list[-1])

                if isinstance(source_config, list):
                    assert (source_list[-1][0]=='-' and str.isdigit(source_list[-1][1:])) or str.isdigit(source_list[-1]), "list index must be int"
                    source_list[-1] = int(source_list[-1])

                to_config[to_list[-1]] = source_config[source_list[-1]]
            except Exception as e:
                logger.error(f"Can not link from '{source}' to '{to}'")
                raise e

        if not link:
            return
        for (source, to) in link.items():
            if isinstance(to, List):
                for sub_to in to:
                    make_link(source, sub_to)
            else:
                make_link(source, to)

    @classmethod
    def collect_link(cls, config, trace: List=None, all_level_links: Dict=None, level=0):
        """collect move all links in config to top

        only do in the top level of config, collect all level links and return the links with level

        Args:
            config: 
                >>> {
                >>>     "arg1": {
                >>>         "arg11": 2
                >>>         "arg12": 3
                >>>         "_link": {"arg11": "arg12"}
                >>>     }
                >>> }
            all_level_links: TODO
            level: TODO

        Returns: 
            >>> {
            >>>     "arg1": {
            >>>         "arg11": 2
            >>>         "arg12": 3
            >>>     }
            >>>     "_link": {"arg1.arg11": "arg1.arg12"}
            >>> }

        """
        if not trace:
            trace = []
        if not all_level_links:
            all_level_links = {}
            
        if level not in all_level_links:
            all_level_links[level] = {}
        trace_str = ".".join(trace)
        if trace_str:
            # except the top level, all add the '.' suffix
            trace_str = trace_str + '.'
        def add_trace(origin_link: Dict)->Dict:
            """add the root of the config to current config trace to current level para of links

            Args:
                origin_link: which is not added the trace(root to cur node)

            Returns: added trace link

            """
            added_trace_link = {}
            for source, target in origin_link.items():
                if isinstance(target, list):
                    target = [trace_str+t for t in target]
                else:
                    target = trace_str + target
                source = trace_str + source
                added_trace_link[source] = target
            return added_trace_link

        if "_link" in config:
            # all_level_links[level] = add_trace(config['_link'])
            all_level_links[level].update(add_trace(config.pop('_link')))

        for submodule_name, submodule_config in config.items():
            if isinstance(submodule_config, dict):
                cls.collect_link(submodule_config, trace + [submodule_name], all_level_links, level+1)
        return all_level_links

    def parser_with_check(self, parser_link=True)->List[Dict]:
        """parser the config and check the config is valid

        Args:
            parser_link: whether parser the links

        Returns: all valided configs

        """
        configs = self.parser(parser_link)
        self.check_config(configs)
        return configs

    def parser(self, parser_link=True) -> List:
        """parser the config

        Args:
            parser_link: whether parser the links

        Returns: all valided configs

        """
        if self.config_file == '*@*':
            return ['*@*']

        # parser submodules get submodules config
        modules_config = self.map_to_submodule(self.modules, self.get_kind_module_base_config)

        # expand all submodules to combine a set of module configs
        possible_config_list = self.get_named_list_cartesian_prod(modules_config)

        # using specifical module config to update base_config
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
        if parser_link:
            for possible_config in all_possible_config_list:
                all_level_links = self.collect_link(possible_config)
                link_union = LinkUnionTool()
                for i in range(len(all_level_links)):
                    cur_level_links = all_level_links[i]
                    link_union.register_low_links(cur_level_links)

                self.config_link_para(link_union.get_links(), possible_config)

        return_list = []
        for possible_config in all_possible_config_list:
            config = copy.deepcopy(possible_config)
            if self.config_name:
                config['_name'] = self.config_name
            return_list.append(config)

        if self.is_rep_config(return_list):
            logger.warning(f"The Configures is Repeated, Please Check The Configures Carefully.")
            for i, config in enumerate(return_list):
                logger.info(f"The {i}th Configure is:")
                logger.info(json.dumps(config, indent=2, ensure_ascii=False))
            raise ValueError('REPEAT CONFIG')
        return return_list

    def get_kind_module_base_config(self, abstract_config: Union[dict, str], kind_module: str="") -> List[dict]:
        """get the whole config of 'kind_module' by given abstract_config

        Args:
            abstract_config: will expanded config
            kind_module: the module kind, like 'embedding', 'subprocessor', which registed in config_parser_register

        Returns: parserd config (whole config) of abstract_config

        """
        return config_parser_register.get(kind_module)(abstract_config).parser(parser_link=False)

    def map_to_submodule(self, config: dict, map_fun: Callable) -> Dict:
        """map the map_fun to all submodules in config

        use the map_fun to process all the modules

        Args:
            config: a dict of submodules, the key is the module kind wich registed in config_parser_register
            map_fun: use the map_fun process the submodule

        Returns: TODO

        """
        modules_config = {}
        for kind_module in config:
            modules_config[kind_module] = map_fun(config[kind_module], kind_module)
        return modules_config

    def load_hjson_file(self, file_path: str) -> Dict:
        """load hjson file from file_path and return a Dict

        Args:
            file_path: the file path

        Returns: loaded dict

        """
        json_file = hjson.load(open(file_path), object_pairs_hook=dict)
        return json_file

    @classmethod
    def flat_search(cls, search, config: dict) -> List[dict]:
        """flat all the _search paras to list

        support recursive parser _search now, this means you can add _search/_link/_base paras in _search paras

        Args:
            search: search paras, {"para1": [1,2,3], 'para2': 'list(range(10))'}
            config: base config

        Returns: list of possible config

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
                extend_config = cls(base_config).parser(parser_link=False)
                result.extend(extend_config)
                # result.append(base_config)

        return result

    def get_cartesian_prod(self, list_of_list_of_dict: List[List[Dict]]) -> List[List[Dict]]:
        """get catesian prod from two lists

        Args:
            list_of_list_of_dict: [[config_a1, config_a2], [config_b1, config_b2]]

        Returns: 
            [[config_a1, config_b1], [config_a1, config_b2], [config_a2, config_b1], [config_a2, config_b2]]

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
        """check all config is valid.

        check all "*@*" is replaced to correct value.
        Args:
            configs: TODO

        Returns: 
            None

        Raises:
            ValueError

        """
        def _check(config):
            """check the "*@*" is in config or not
            """
            for key in config:
                if isinstance(config[key], dict):
                    _check(config[key])
                if config[key] == '*@*':
                    raise ValueError(f'In Config: \n {json.dumps(config, indent=4, ensure_ascii=False)}\n The must be provided key "{key}" marked with "*@*" is not provided.')

        if isinstance(configs, list):
            for config in configs:
                _check(config)
        else:
            _check(configs)

    @staticmethod
    def get_named_list_cartesian_prod(dict_of_list: Dict[str, List]=None) -> List[Dict]:
        """get catesian prod from named lists

        Args:
            dict_of_list: {'name1': [1,2,3], 'name2': [1,2,3]}

        Returns: 
            [{'name1': 1, 'name2': 1}, {'name1': 1, 'name2': 2}, {'name1': 1, 'name2': 3}, ...]

        """
        if not dict_of_list:
            dict_of_list = {}
        if len(dict_of_list) == 0:
            return []
        dict_of_list = copy.deepcopy(dict_of_list)
        cur_name, cur_paras  = dict_of_list.popitem()
        cur_para_search_list = []
        if isinstance(cur_paras, str):
            cur_paras = eval(cur_paras)
        assert isinstance(cur_paras, list), f"The search options must be list, but you provide {cur_paras}({type(cur_paras)})"
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

        Args:
            list_of_dict: a list of dict

        Returns:
            has repeat or not

        """
        # using json.dumps + sort_keys to guarantee the same dict to the same string represatation
        list_of_str = [json.dumps(dic, sort_keys=True, ensure_ascii=False) for dic in list_of_dict]
        if len(list_of_dict) == len(set(list_of_str)):
            return False
        else:
            return True


@config_parser_register('config')
class ConfigConfigParser(BaseConfigParser):
    """ConfigConfigParser"""
    def __init__(self, config_file):
        super(ConfigConfigParser, self).__init__(config_file, config_base_dir='NONEPATH')
        if self.base_config:
            raise AttributeError('The paras config do not support _base.')

        if self.config_name:
            raise AttributeError('The paras config do not support _name.')

    def parser(self, parser_link=True):
        """parser the config

        config support _search and _link

        Args:
            parser_link: whether parser the links

        Returns:
            all valided configs

        """
        config_list = self.flat_search(self.search, self.modules)
        # link paras
        if parser_link:
            for possible_config in config_list:
                all_level_links = self.collect_link(possible_config)
                link_union = LinkUnionTool()
                for i in range(len(all_level_links)):
                    cur_level_links = all_level_links[i]
                    link_union.register_low_links(cur_level_links)
                self.config_link_para(all_level_links, possible_config)

        return config_list


@config_parser_register('_link')
class LinkConfigParser(object):
    """LinkConfigParser"""
    def __init__(self, config_file):
        self.config = config_file
        assert isinstance(self.config, dict), f"The '_link' must be a dict, but you provide '{self.config}'"
    def parser(self, parser_link=False):
        """parser the config

        config support _search and _link

        Args:
            parser_link: must be false

        Returns:
            all valided configs

        """
        assert parser_link is False, f"The parser_link para must be False when parser the _link"
        return [self.config]


@config_parser_register('task')
class TaskConfigParser(BaseConfigParser):
    """docstring for TaskConfigParser"""
    def __init__(self, config_file):
        super(TaskConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/tasks/')


@config_parser_register('root')
class RootConfigParser(BaseConfigParser):
    """docstring for RootConfigParser"""
    def __init__(self, config_file):
        super(RootConfigParser, self).__init__(config_file, config_base_dir='')


@config_parser_register('manager')
class ManagerConfigParser(BaseConfigParser):
    """docstring for ManagerConfigParser"""
    def __init__(self, config_file):
        super(ManagerConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/managers/', register=manager_config_register)


@config_parser_register('callback')
class CallbackConfigParser(BaseConfigParser):
    """docstring for CallbackConfigParser"""
    def __init__(self, config_file):
        super(CallbackConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/callbacks/', register=callback_config_register)


@config_parser_register('datamodule')
class DatamoduleConfigParser(BaseConfigParser):
    """docstring for DatamoduleConfigParser"""
    def __init__(self, config_file):
        super(DatamoduleConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/data/datamodules/', register=datamodule_config_register)


@config_parser_register('imodel')
class IModelConfigParser(BaseConfigParser):
    """docstring for IModelConfigParser"""
    def __init__(self, config_file):
        super(IModelConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/imodels/', register=imodel_config_register)


@config_parser_register('model')
class ModelConfigParser(BaseConfigParser):
    """docstring for ModelConfigParser"""
    def __init__(self, config_file):
        super(ModelConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/models/', register=model_config_register)


@config_parser_register('optimizer')
class OptimizerConfigParser(BaseConfigParser):
    """docstring for OptimizerConfigParser"""
    def __init__(self, config_file):
        super(OptimizerConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/optimizers/', register=optimizer_config_register)


@config_parser_register('scheduler')
class ScheduleConfigParser(BaseConfigParser):
    """docstring for ScheduleConfigParser"""
    def __init__(self, config_file):
        super(ScheduleConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/schedulers/', register=scheduler_config_register)

@config_parser_register('initmethod')
class InitMethodConfigParser(BaseConfigParser):
    """docstring for InitMethodConfigParser"""
    def __init__(self, config_file):
        super(InitMethodConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/initmethods/', register=initmethod_config_register)


@config_parser_register('loss')
class LossConfigParser(BaseConfigParser):
    """docstring for LossConfigParser"""
    def __init__(self, config_file):
        super(LossConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/losses/', register=loss_config_register)


@config_parser_register('encoder')
class EncoderConfigParser(BaseConfigParser):
    """docstring for EncoderConfigParser"""
    def __init__(self, config_file):
        super(EncoderConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/layers/encoders/', register=encoder_config_register)


@config_parser_register('decoder')
class DecoderConfigParser(BaseConfigParser):
    """docstring for DecoderConfigParser"""
    def __init__(self, config_file):
        super(DecoderConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/layers/decoders/', register=decoder_config_register)


@config_parser_register('embedding')
class EmbeddingConfigParser(BaseConfigParser):
    """docstring for EmbeddingConfigParser"""
    def __init__(self, config_file):
        super(EmbeddingConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/layers/embeddings/', register=embedding_config_register)


@config_parser_register('module')
class ModuleConfigParser(BaseConfigParser):
    """docstring for ModuleConfigParser"""
    def __init__(self, config_file):
        super(ModuleConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/core/modules/', register=module_config_register)

@config_parser_register('processor')
class ProcessorConfigParser(BaseConfigParser):
    """docstring for ProcessorConfigParser"""
    def __init__(self, config_file):
        super(ProcessorConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/data/processors/', register=processor_config_register)


@config_parser_register('subprocessor')
class SubProcessorConfigParser(BaseConfigParser):
    """docstring for SubProcessorConfigParser"""
    def __init__(self, config_file):
        super(SubProcessorConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/data/subprocessors/', register=subprocessor_config_register)


@config_parser_register('postprocessor')
class PostProcessorConfigParser(BaseConfigParser):
    """docstring for PostProcessorConfigParser"""
    def __init__(self, config_file):
        super(PostProcessorConfigParser, self).__init__(config_file, config_base_dir='dlk/configures/data/postprocessors/', register=postprocessor_config_register)

