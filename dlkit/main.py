import hjson
import os
from typing import Dict, Union, Callable, List, Any
# from models import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY
from dlkit.utils.parser import CONFIG_PARSER_REGISTRY 
import json


class Train(object):
    """docstring for Train"""
    def __init__(self, config_file):
        super(Train, self).__init__()
        self.config_file = self.load_hjson_file(config_file)
        self.focus = self.config_file.pop('_focus', {})
        parser = CONFIG_PARSER_REGISTRY['task'](self.config_file)
        self.configs = parser.parser()
        self.config_names = []
        for possible_config in self.configs:
            config_name = []
            for source, to in self.focus.items():
                config_point = possible_config
                trace = source.split('.')
                for t in trace:
                    config_point = config_point[t]
                config_name.append(to+"="+str(config_point))
            self.config_names.append('_'.join(config_name))

        if len(self.config_names) != len(set(self.config_names)):
            for config, name in zip(self.configs, self.config_names):
                print(json.dumps(config, indent=4))
                print(name)
            raise NameError('The config_names is not unique.')
        for config, name in zip(self.configs, self.config_names):
            print(json.dumps(config, indent=4))
            print(name)


    def load_hjson_file(self, file_name: str) -> Dict:
        """load hjson file by file_name

        :file_name: TODO
        :returns: TODO

        """
        json_file = hjson.load(open(file_name), object_pairs_hook=dict)
        return json_file

# Train('simple_ner')
Train('./configures/tasks/simple_ner.hjson')
# Train('lstm_linear_ner')
# Train('test')
# Train('lstm')
# print(task_config)


