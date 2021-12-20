"""
Loader the $meta, etc. to data
"""
from dlk.utils.config import ConfigTool, BaseConfig
from dlk.utils.logger import Logger
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
import pickle as pkl
import os

logger = Logger.get_logger()

@subprocessor_config_register('load')
class LoadConfig(BaseConfig):
    """
    Config eg.
     {
         "_name": "load",
         "config":{
             "base_dir": "."
             "predict":{
                 "meta": "./meta.pkl",
             },
             "online": [
                 "predict", //base predict
                 {   // special config, update predict, is this case, the config is null, means use all config from "predict", when this is empty dict, you can only set the value to a str "predict", they will get the same result
                 }
             ]
         }
     },
    """

    def __init__(self, stage, config):
        super(LoadConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.base_dir:str = config.get('config').get("base_dir", ".")


@subprocessor_register('load')
class Load(ISubProcessor):
    """
    """

    def __init__(self, stage: str, config: LoadConfig):
        super().__init__()
        self.stage = stage
        self.config = config.config
        if not self.config:
            logger.info(f"Skip 'load' at stage {self.stage}")
            return
        self.base_dir = config.base_dir

    def load(self, path):
        """TODO: Docstring for load.
        """

        logger.info(f"Loading file from {os.path.join(self.base_dir, path)}")
        return pkl.load(open(os.path.join(self.base_dir, path), 'rb'))

    def process(self, data: Dict)->Dict:
        for _, path in self.config.items():
            meta = self.load(path)
            for key, value in meta.items():
                data[key] = value
        return data
