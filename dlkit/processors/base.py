import hjson
import pandas as pd
import sys
from typing import Union
sys.path.append('../../')
from dlkit.utils.parser import BaseConfigParser
from pprint import pprint

# json = hjson.load(open('./process.hjson'), object_pairs_hook=dict)
# link = json.pop("_link", {})
# BaseConfigParser.config_link_para(link, json)
# pprint(json)
# print(int('-1'))

class Process(object):
    """docstring for DataSet"""
    def __init__(self, config: Union[dict, str], stage: str):
        super(Process, self).__init__()
        if isinstance(config, str):
            config = hjson.load(open(config), object_pairs_hook=dict)
        BaseConfigParser.config_link_para(link, config)
        self.config = config


    def process_instance(self, origin_instance: dict):
        """TODO: Docstring for process.

        :origin_instance: TODO
        :returns: TODO

        """
        pass

    def process(self, data: pd.DataFrame)->pd.DataFrame:
        """TODO: Docstring for process.

        :data: TODO
        :returns: TODO

        """
        pass
        return data
