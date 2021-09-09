import hjson
import os
from typing import Dict, Union, Callable, List, Any
# from models import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY
from utils.config import ModelJsonConfigParser, TokenizerJsonConfigParser, OptimizerJsonConfigParser, DataloaderJsonConfigParser, ProcessorJsonConfigParser

config_dir_parser_map = {
    'model': {
        'path': 'configures/models/',
        'parser': ModelJsonConfigParser
    },
    'dataloader': {
        'path': 'configures/dataloaders/',
        'parser': DataloaderJsonConfigParser
    },
    'optimizer': {
        'path': 'configures/optimizers/',
        'parser': OptimizerJsonConfigParser
    },
    'processor': {
        'path': 'configures/processors/',
        'parser': ProcessorJsonConfigParser
    },
}


class Train(object):
    """docstring for Train"""
    def __init__(self, config_file):
        super(Train, self).__init__()
        self.config_file = config_file
        self.task_config = self.load_hjson_file(config_file)
        self.task_name = self.task_config.get('task', None)
        self.check_config_para_not_null(self.task_name, "task name")

        # self.processor_config = self.parser_config(self.task_config.get('processor', ''), 'processor config', 'processor')

        self.model_config = self.parser_config(self.task_config.get('model', ''), 'model config', 'model')
        for config in self.model_config:
            print(config)
        # self.optimizer_config = self.parser_config(self.task_config.get('optimizer', ''), 'optimizer config', 'optimizer')

        # self.dataloader_config = self.parser_config(self.task_config.get('dataloader', ''), 'dataloader config', 'dataloader')

        # self.model_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.model_config, "model config")
        # self.model_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.model_config, "model config")
        # self.dataloader_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.dataloader_config, "dataloader config")
        # self.optimizer_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.optimizer_config, "optimizer config")
    def get_cartesian_prod(self, list_of_dict_a: List[List[Dict]], list_of_dict_b: List[List[Dict]]) -> List[List[Dict]]:
        """get catesian prod from two lists
        :list_of_dict_a: [[config_a1], [config_a2]]
        :list_of_dict_b: [[config_b1, config_c1], [config_b1, config_c2], [config_b2, config_c1], [config_b2, config_c2]]
        :returns: [[config_a1, config_b1, config_c1], 
                   [config_a1, config_b1, config_c2], 
                   [config_a1, config_b1, config_c3], 
                   [config_a1, config_b2, config_c1], 
                   [config_a1, config_b2, config_c2], 
                   [config_a1, config_b2, config_c3], 
                   ....
                   [config_a3, config_b3, config_c3], 
                  ]
        """
        if len(list_of_dict_b) == 0:
            return list_of_dict_a
        result = []
        for cur_config in list_of_dict_a:
            for reserve in list_of_dict_b:
                result.append(cur_config+reserve)

        return result

    def parser_config(self, config: Union[str, Dict, List], para_name: str, config_type: str)->List[List[Dict]]:
        """TODO: Docstring for parser_config.

        :config: Union[str, Dict, List]
        :para_name: str TODO
        :config_type: str TODO
        :returns: List[List[Dict]] for multiple module and search para

        """
        if not config:
            return []
        config_name, extend_config = "", {}
        if isinstance(config, str):
            config_name = config
            extend_config = {}
        elif isinstance(config, dict):
            config_name = config.pop('name')
            self.check_config_para_not_null(config_name, para_name+' name')
            extend_config = config
        elif isinstance(config, list):
            return self.get_cartesian_prod(
                self.parser_config(config[0], para_name, config_type), self.parser_config(config[1:], para_name, config_type))
        else:
            raise KeyError('The configure of {} must be string or dict or list.'.format(para_name))
        config_path = os.path.join(config_dir_parser_map[config_type]['path'], config_name+'.hjson')
        config_list = config_dir_parser_map[config_type]['parser'](config_path).parser(extend_config)

        return [[dic] for dic in config_list]

    def load_hjson_file(self, file_name: str) -> Dict:
        """load hjson file by file_name

        :file_name: TODO
        :returns: TODO

        """
        json_file = hjson.load(open(file_name), object_pairs_hook=dict)
        return json_file

    def check_config_para_not_null(self, para: Any, para_name: str) -> None:
        """TODO: Docstring for check_not_null.

        :arg1: TODO
        :returns: TODO

        """
        if not para:
            raise KeyError("You must provide the {} in {}".format(para_name, self.config_file))

Train('./configures/tasks/simple_ner.hjson')
# print(task_config)
