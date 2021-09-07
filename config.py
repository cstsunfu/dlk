import hjson
import os
import copy
from typing import Callable, List, Dict, Tuple

        
class BaseConfigParser(object):
    """docstring for BaseConfigParser"""
    def __init__(self, config_file_path, configure_base_path):
        super(BaseConfigParser, self).__init__()
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
        

class TokenizerConfigParser(BaseConfigParser):
    """docstring for ModelConfigParser"""
    def __init__(self, config_file_path, configure_base_path='configures/tokenizers'):
        super(TokenizerConfigParser, self).__init__(config_file_path=config_file_path, configure_base_path=configure_base_path)
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


class ModelConfigParser(BaseConfigParser):
    """docstring for ModelConfigParser"""
    def __init__(self, config_file_path, configure_base_path='configures/modules'):
        super(ModelConfigParser, self).__init__(config_file_path=config_file_path, configure_base_path=configure_base_path)

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



# class ModelConfigParser(object):
    # """docstring for ModelConfigParser"""
    # def __init__(self, config_file_path):
        # super(ModelConfigParser, self).__init__()
        # self.config_file_path = config_file_path
        # self.model_config = self.load_hjson_file(self.config_file_path)
        # self.config_name = self.model_config['name']
        # self.config = self.model_config['config']
        # self.configure_base_path = 'configures/modules'
        
    # def _update_config(self, config: dict, update_config) ->Dict:
        # """use update_config update the config

        # :config: will updated config
        # :returns: updated config

        # """
        # for module in update_config:
            # if module not in config:
                # raise KeyError('The model config has not the module "{}"'.format(module))
            # config[module]['config'].update(update_config[module])

        # return config

    # def parser(self, update_config: dict={}) -> List[Dict]:
        # """return a list of para dict s for all possible combinations
        # :update_config: the update_config will update the returned config
        # :returns: list[possible config dict]

        # """


        # all_module_config_in_one = self._parser(self.config)
        # all_module_config_list = self.get_cartesian_prod(all_module_config_in_one)
        # return_list = []
        # for config in all_module_config_list:
            # config = self._update_config(config, update_config)

            # return_list.append({
                # 'name': self.config_name,
                # 'config': config
                # })
        # return return_list

    # def load_hjson_file(self, file_name: str) -> Dict:
        # """load hjson file by file_name

        # :file_name: TODO
        # :returns: TODO

        # """
        # json_file = hjson.load(open(file_name), object_pairs_hook=dict)
        # return json_file

    # def update_base_flat_search(self, abstract_config: dict, kind_module: str='') -> List[dict]:
        # """update the __base__ config from file by new config, and extend all __search__ para to a completely config and form a list

        # :abstract_config: dict: {'__base__': base module name, '__search__': {'para_name': [serch list], ...}, 'config': {the new config to update base config}}
        # :kind_module: str: module kind name
        # :returns result: List[dict]: [{'name': 'module name', config: {module config}}, ...]
        # """
        # result = []
        # module_config = abstract_config.get('config', {})
        # module_base_config_name = abstract_config.get('__base__', None) # configure name the path to configure/modules/kind_module/module_base_config_name
        # if not module_base_config_name:
            # raise KeyError('You must provide the __base__ module in configure {}'.format(kind_module))
        # module_base_config = self.load_hjson_file(os.path.join(self.configure_base_path, kind_module+'s', module_base_config_name+'.hjson'))
        # module_search_para = abstract_config.get('__search__', None)
        # module_base_config['config'].update(module_config)
        # module_base_name = module_base_config['name'] # module name like "lstm、linear、crf、etc."
        # if not module_search_para:
            # result.append({
                # 'name': module_base_name,
                # 'config': module_base_config['config']
            # })
        # else:
            # search_para_list = self.get_cartesian_prod(module_search_para)
            # for search_para in search_para_list:
                # base_config = copy.deepcopy(module_base_config['config'])
                # base_config.update(search_para)
                # result.append({"name": module_base_name, "config": base_config})

        # return result

    # def _parser(self, config):
        # """join base configure and user defined config, if there is search para return all para 
        # :returns: {'module_name1': [first_possible_para_dict, second_possible_para_dict], 'module_name2': [first_possible_para_dict, second_possible_para_dict], ...}

        # """
        # all_module_config_in_one = {}

        # for kind_module in config:
            # all_module_config_in_one[kind_module] = self.update_base_flat_search(config[kind_module], kind_module)
        # return all_module_config_in_one

    # def get_cartesian_prod(self, dict_of_list: Dict[str, List]) -> List[Dict]:
        # """get catesian prod from named lists
        # :dict_of_list: {'name1': [1,2,3], 'name2': [1,2,3]}
        # :returns: [{'name1': 1, 'name2': 1}, {'name1': 1, 'name2': 2}, {'name1': 1, 'name2': 3}, ...]

        # """
        # if len(dict_of_list) == 0:
            # return []
        # cur_name, cur_paras  = dict_of_list.popitem()
        # cur_para_search_list = []
        # for para in cur_paras:
            # cur_para_search_list.append({cur_name: para})
        # if len(dict_of_list) == 0:
            # return cur_para_search_list
        # reserve_para_list = self.get_cartesian_prod(dict_of_list)
        # all_config_list = []
        # for cur_config in cur_para_search_list:
            # for reserve_config in reserve_para_list:
                # cur_config = copy.deepcopy(cur_config)
                # cur_config.update(reserve_config)
                # all_config_list.append(cur_config)

        # return all_config_list


path = './configures/models/lstm_linear_ner.hjson'

# path = './configures/tokenizers/bpe.hjson'
# path = './tests/token.hjson'
parser = ModelConfigParser(path)
# parser = TokenizerConfigParser(path)
for i in parser.parser(update_config={'encoder':{"config":{'output_size': False}, "__search__":{"bidirection": [True, False]}}}):
# for i in parser.parser(update_config={'config':{"model_path":''}}):
    print(i)
