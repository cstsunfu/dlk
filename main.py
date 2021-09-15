import hjson
import os
from typing import Dict, Union, Callable, List, Any
# from models import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY
from utils.parser import CONFIG_PARSER_REGISTRY 


class Train(object):
    """docstring for Train"""
    def __init__(self, config_file):
        super(Train, self).__init__()
        self.config_file = self.load_hjson_file(config_file)
        self.focus = self.config_file.pop('__focus__', {})
        parser = CONFIG_PARSER_REGISTRY['system'](self.config_file)
        config = parser.parser()
        print(config)

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
