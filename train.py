import hjson
import os
from typing import Dict, Union, Callable, List, Any
# from models import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY
from utils.config import ModelConfigParser, TokenizerConfigParser, OptimizerConfigParser, DatasetConfigParser

config_dir_parser_map = {
    'model': {
        'path': 'configures/models/',
        'parser': ModelConfigParser
    },
    'dataset': {
        'path': 'configures/datasets/',
        'parser': DatasetConfigParser
    },
    'optimizer': {
        'path': 'configures/optimizers/',
        'parser': OptimizerConfigParser
    }
}


class Train(object):
    """docstring for Train"""
    def __init__(self, config_file):
        super(Train, self).__init__()
        self.config_file = config_file
        self.task_config = self.load_hjson_file(config_file)
        self.task_name = self.task_config.get('task', None)
        self.check_config_para_not_null(self.task_name, "task name")

        self.model_config = self.parser_config(self.task_config.get('model', ''), 'model config', 'model')
        for config in self.model_config:
            print(config)

        # self.model_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.model_config, "model config")
        # self.model_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.model_config, "model config")
        # self.dataset_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.dataset_config, "dataset config")
        # self.optimizer_config = self.task_config.get('model', None)
        # self.check_config_para_not_null(self.optimizer_config, "optimizer config")

    def parser_config(self, config: Union[str, Dict], para_name: str, config_type: str)->List[Dict]:
        """TODO: Docstring for parser_config.

        :config: Union[str, Dict]
        :para_name: str TODO
        :config_type: str TODO
        :returns: TODO

        """
        self.check_config_para_not_null(config, para_name)
        config_name, extend_config = "", {}
        if isinstance(config, str):
            config_name = config
            extend_config = {}
        elif isinstance(config, dict):
            config_name = config.pop('name')
            self.check_config_para_not_null(config_name, para_name+' name')
            extend_config = config
        else:
            raise KeyError('The configure of {} must be string or dict.'.format(para_name))
        config_path = os.path.join(config_dir_parser_map[config_type]['path'], config_name+'.hjson')
        config_list = config_dir_parser_map[config_type]['parser'](config_path).parser(extend_config)

        return config_list

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
