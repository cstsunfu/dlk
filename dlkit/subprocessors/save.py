from dlkit.utils.config import Config, GetConfigByStageMixin
from typing import Dict, Callable, Set, List
from dlkit.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
import pickle as pkl
import copy
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
                "processed": "processed_data.pkl", // all data
                "meta": {
                    "meta.pkl": ['label_ids', 'embedding'] //only for next time use
                }
            },
            "predict": {
                "processed": "processed_data.pkl",
            }
        }
    },
    """

    def __init__(self, stage, config):
        self.config = self.get_config(stage, config)
        self.base_dir:str = config.get('config').get("base_dir", ".")

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
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        return pkl.dump(data, open(os.path.join(self.base_dir, path), 'wb'))

    def process(self, data: Dict)->Dict:
        if not self.config:
            return data
        if "processed" in self.config:
            self.save(data, self.config['processed'])
        if "meta" in self.config:
            for save_path, save_fileds in self.config['meta'].items():
                assert isinstance(save_fileds, list)
                meta_data = {}
                for field in save_fileds:
                    meta_data[field] = copy.deepcopy(data[field])
                self.save(meta_data, save_path)
        return data
