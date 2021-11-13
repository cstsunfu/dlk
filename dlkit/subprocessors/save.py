from dlkit.utils.config import Config, GetConfigByStageMixin
from typing import Dict, Callable, Set, List
from dlkit.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
import pickle as pkl
import os


@subprocessor_config_register('save')
class SaveConfig(Config, GetConfigByStageMixin):
    """
    Config eg.
    {
        "_name": "save",
        "config":{
            "base_dir": "."
            "train":{
                "data.train": "./train.pkl",
                "data.dev": "./dev.pkl",
                "token_ids": "./token_ids.pkl",
                "embedding": "./embedding.pkl",
                "label_ids": "./label_ids.pkl",
            },
            "predict": {
                "data.predict": "./predict.pkl"
            }
        }
    },
    """

    def __init__(self, stage, config):
        self.config = self.get_config(stage, config)
        self.base_dir:str = config.get("base_dir", ".")

@subprocessor_register('save')
class Save(ISubProcessor):
    """
    """

    def __init__(self, stage: str, config: SaveConfig):
        super().__init__()
        self.stage = stage
        self.config = config.config
        self.base_dir = config.base_dir

    def save(self, data, path):
        """TODO: Docstring for load.
        """
        return pkl.dump(data, open(os.path.join(self.base_dir, path), 'wb'))

    def process(self, data: Dict)->Dict:
        for key, value in self.config.items():
            subkeys = key.split('.')
            _data = data
            for subkey in subkeys:
                _data = _data[subkey]
            self.save(_data, value)
        return data
