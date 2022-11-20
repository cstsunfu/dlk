# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dlk.utils.config import ConfigTool, BaseConfig
from dlk.utils.logger import Logger
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
import pickle as pkl
from dlk.utils.io import open
import copy
import os

logger = Logger.get_logger()


@subprocessor_config_register('save')
class SaveConfig(BaseConfig):
    default_config = {
        "_name": "save",
        "config":{
            "base_dir": "",
            "train":{
                "processed": "processed_data.pkl", # all data without meta
                "meta": "*@*"
            },
            "predict": {
                "processed": "processed_data.pkl",
            },
            "extend_train": {
                "processed": "processed_data_extend.pkl",
            },
        }
    }
    """Config for Save
    Config Example:
        default_config
    """
    def __init__(self, stage, config):
        super(SaveConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.base_dir: str = config.get('config').get("base_dir", "")
        self.post_check(self.config, used=[
            "processed",
            "meta",
        ])


@subprocessor_register('save')
class Save(ISubProcessor):
    """
    Save the processed data to $base_dir/$processed
    Save the meta data(like vocab, embedding, etc.) to $base_dir/$meta
    """
    def __init__(self, stage: str, config: SaveConfig):
        super().__init__()
        self.stage = stage
        self.config = config.config
        if not self.config:
            logger.info(f"Skip 'save' at stage {self.stage}")
            return
        self.base_dir = config.base_dir

    def save(self, data, path: str):
        """save data to path

        Args:
            data: pickleable data
            path: the path to data

        Returns: 
            loaded data

        """
        logger.info(f"Saving file to {os.path.join(self.base_dir, path)}")
        with open(os.path.join(self.base_dir, path), 'wb') as f:
            return pkl.dump(data, f)

    def process(self, data: Dict) -> Dict:
        """Save entry

        Args:
            data: 
            >>> {
            >>>     "data": {"train": ...},
            >>>     "tokenizer": ..
            >>> }

        Returns: 
            data

        """
        if not self.config:
            return data
        # record for save only in meta, not in processed_data but returned all
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
