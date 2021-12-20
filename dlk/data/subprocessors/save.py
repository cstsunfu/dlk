"""
Save the processed data to $base_dir/$processed
Save the meta data(like vocab, embedding, etc.) to $base_dir/$meta
"""
from dlk.utils.config import ConfigTool, BaseConfig
from dlk.utils.logger import Logger
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
import pickle as pkl
import copy
import os

logger = Logger.get_logger()

@subprocessor_config_register('save')
class SaveConfig(BaseConfig):
    """
    Config eg.
    {
        "_name": "save",
        "config":{
            "base_dir": "."
            "train":{
                "processed": "processed_data.pkl", // all data without meta
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
        super(SaveConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.base_dir:str = config.get('config').get("base_dir", ".")
        self.post_check(self.config, used=[
            "processed",
            "meta",
        ])

@subprocessor_register('save')
class Save(ISubProcessor):
    """
    """

    def __init__(self, stage: str, config: SaveConfig):
        super().__init__()
        self.stage = stage
        self.config = config.config
        if not self.config:
            logger.info(f"Skip 'save' at stage {self.stage}")
            return
        self.base_dir = config.base_dir

    def save(self, data, path):
        """TODO: Docstring for load.
        """
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        logger.info(f"Saving file to {os.path.join(self.base_dir, path)}")
        return pkl.dump(data, open(os.path.join(self.base_dir, path), 'wb'))

    def process(self, data: Dict)->Dict:
        if not self.config:
            return data
        meta_fileds = set()
        if "meta" in self.config:
            for save_path, save_fileds in self.config['meta'].items():
                assert isinstance(save_fileds, list)
                meta_data = {}
                for field in save_fileds:
                    meta_data[field] = copy.deepcopy(data[field])
                    meta_fileds.add(field)
                self.save(meta_data, save_path)
        reserve_meta_map = {}
        for filed in meta_fileds:
            reserve_meta_map[filed] = data.pop(filed)
        if "processed" in self.config:
            self.save(data, self.config['processed'])
        data.update(reserve_meta_map)
        return data
