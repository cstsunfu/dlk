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

from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
import pandas as pd
from dlk.utils.logger import Logger

logger = Logger.get_logger()

@subprocessor_config_register('seq_lab_loader')
class SeqLabLoaderConfig(BaseConfig):
    """Config for SeqLabLoader

    Config Example:
        >>> {
        >>>     "_name": "seq_lab_loader",
        >>>     "config": {
        >>>         "train":{ //train、predict、online stage config,  using '&' split all stages
        >>>             "data_set": {                   // for different stage, this processor will process different part of data
        >>>                 "train": ['train', 'valid', 'test', 'predict'],
        >>>                 "predict": ['predict'],
        >>>                 "online": ['online']
        >>>             },
        >>>             "input_map": {   // without necessery don't change this
        >>>                 "sentence": "sentence",
        >>>                 "uuid": "uuid",
        >>>                 "entities_info": "entities_info",
        >>>             },
        >>>             "output_map": {   // without necessery don't change this
        >>>                 "sentence": "sentence",
        >>>                 "uuid": "uuid",
        >>>                 "entities_info": "entities_info",
        >>>             },
        >>>         }, //3
        >>>         "predict": "train",
        >>>         "online": "train",
        >>>     }
        >>> }
    """
    def __init__(self, stage, config: Dict):

        super(SeqLabLoaderConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.output_map = self.config.get('output_map', {})
        self.input_map = self.config.get('input_map', {})
        self.post_check(self.config, used=[
            "data_set",
            "input_map",
            "output_map",
        ])


@subprocessor_register('seq_lab_loader')
class SeqLabLoader(ISubProcessor):
    """
    Loader the data from dict and generator DataFrame
    """

    def __init__(self, stage: str, config: SeqLabLoaderConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'seq_lab_loader' at stage {self.stage}")
            return

    def process(self, data: Dict)->Dict:
        """Entry for sequence labeling loader

        Args:
            data: 
                >>> input data
                >>> {
                >>>     "train": list of json format train data
                >>> }
                >>> one_ins example:
                >>> {
                >>>     "uuid": '**-**-**-**'
                >>>     "sentence": "I have an apple",
                >>>     "labels": [
                >>>                 {
                >>>                     "end": 15,
                >>>                     "start": 10,
                >>>                     "labels": [
                >>>                         "Fruit"
                >>>                     ]
                >>>                 },
                >>>                 ...,
                >>>             ]
                >>>         },
                >>>     ],
                >>> }
        Returns: 
            data + loaded_data
        """

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do seq_lab_loader on it.')
                continue
            data_set = data['data'][data_set_name]

            sentences = []
            uuids = []
            entities_infos = []

            for one_ins in data_set:
                try:
                    sentences.append(one_ins[self.config.input_map['sentence']])
                    uuids.append(one_ins[self.config.input_map['uuid']])
                    entities_infos.append(one_ins[self.config.input_map['entities_info']])
                except:
                    raise PermissionError(f"You must provide the data as requests, we need 'sentence', 'uuid' and 'entities_info', or you can provide the input_map to map the origin data to this format")
            data_df = pd.DataFrame(data= {
                self.config.output_map["sentence"]: sentences,
                self.config.output_map["uuid"]: uuids,
                self.config.output_map["entities_info"]: entities_infos,
            })
            data['data'][data_set_name] = data_df

        return data
