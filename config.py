import hjson
import os
import copy
from typing import List, Dict


class ModelConfigParser(object):
    """docstring for ModelConfigParser"""
    def __init__(self, config_name):
        super(ModelConfigParser, self).__init__()
        self.config_name = config_name
        self.model_config = self.load_hjson_file(self.config_name)
        self.config_name = self.model_config['name']
        self.config = self.model_config['config']
        self.configure_base_path = 'configure/modules'
        
    def parser(self) -> List[Dict]:
        """return a list of para dict s for all possible combinations
        :returns: list[possible config dict]

        """
        all_module_config_in_one = self._parser(self.config)
        all_module_config_list = self.get_cartesian_prod(all_module_config_in_one)
        return_list = []
        for config in all_module_config_list:
            return_list.append({
                'name': self.config_name,
                'config': config
                })
        return return_list

    def load_hjson_file(self, file_name: str) -> Dict:
        """load hjson file by file_name

        :file_name: TODO
        :returns: TODO

        """
        json_file = hjson.load(open(file_name), object_pairs_hook=dict)
        return json_file

    def _parser(self, config):
        """join base configure and user defined config, if there is search para return all para 
        :returns: {'module_name1': [first_possible_para_dict, second_possible_para_dict], 'module_name2': [first_possible_para_dict, second_possible_para_dict], ...}

        """
        all_module_config_in_one = {}

        for kind_module in config:
            all_module_config_in_one[kind_module] = []
            module_config = config[kind_module].get('config', {})
            module_base_config_name = config[kind_module].get('__base__', None) # configure name the path to configure/modules/kind_module/module_base_config_name
            if not module_base_config_name:
                raise KeyError('You must provide the __base__ module in configure {}'.format(kind_module))
            module_base_config = self.load_hjson_file(os.path.join(self.configure_base_path, kind_module+'s', module_base_config_name+'.hjson'))
            module_search_para = config[kind_module].get('__search__', None)
            module_base_config['config'].update(module_config)
            module_base_name = module_base_config['name'] # module name like "lstm、linear、crf、etc."
            if not module_search_para:
                all_module_config_in_one[kind_module].append({
                    'name': module_base_name,
                    'config': module_base_config['config']
                })
            else:
                search_para_list = self.get_cartesian_prod(module_search_para)
                for search_para in search_para_list:
                    base_config = copy.deepcopy(module_base_config['config'])
                    base_config.update(search_para)
                    all_module_config_in_one[kind_module].append({"name": module_base_name, "config": base_config})
        return all_module_config_in_one

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
                cur_config.update(reserve_config)
                all_config_list.append(cur_config)

        return all_config_list
