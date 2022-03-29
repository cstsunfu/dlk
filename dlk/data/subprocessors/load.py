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
import os

logger = Logger.get_logger()

@subprocessor_config_register('load')
class LoadConfig(BaseConfig):
    """Config for Load

    Config Example:
        >>> {
        >>>     "_name": "load",
        >>>     "config":{
        >>>         "base_dir": "."
        >>>         "predict":{
        >>>             "meta": "./meta.pkl",
        >>>         },
        >>>         "online": [
        >>>             "predict", //base predict
        >>>             {   // special config, update predict, is this case, the config is null, means use all config from "predict", when this is empty dict, you can only set the value to a str "predict", they will get the same result
        >>>             }
        >>>         ]
        >>>     }
        >>> },
    """

    def __init__(self, stage, config):
        super(LoadConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.base_dir:str = config.get('config').get("base_dir", ".")


@subprocessor_register('load')
class Load(ISubProcessor):
    """ Loader the $meta, etc. to data
    """

    def __init__(self, stage: str, config: LoadConfig):
        super().__init__()
        self.stage = stage
        self.config = config.config
        self.load_data = {}
        if not self.config:
            logger.info(f"Skip 'load' at stage {self.stage}")
            return
        self.base_dir = config.base_dir

        for key, path in self.config.items():
            self.load_data[key] = self.load(path)

    def load(self, path: str):
        """load data from path

        Args:
            path: the path to data

        Returns: 
            loaded data

        """

        logger.info(f"Loading file from {os.path.join(self.base_dir, path)}")
        return pkl.load(open(os.path.join(self.base_dir, path), 'rb'))

    def process(self, data: Dict)->Dict:
        """Load entry

        Args:
            data: 
            >>> {
            >>>     "data": {"train": ...},
            >>>     "tokenizer": ..
            >>> }

        Returns: 
            data + loaded_data

        """
        for _, meta in self.load_data.items():
            for key, value in meta.items():
                data[key] = value
        return data
