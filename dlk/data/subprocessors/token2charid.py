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
from dlk.utils.logger import Logger
import os

logger = Logger.get_logger()

@subprocessor_config_register('token2charid')
class Token2CharIDConfig(BaseConfig):
    default_config = {
        "_name": "token2charid",
        "config": {
            "train":{
                "data_pair": {
                    "sentence & offsets": "char_ids"
                },
                "data_set": {                   # for different stage, this processor will process different part of data
                    "train": ['train', 'valid', 'test', 'predict'],
                    "predict": ['predict'],
                    "online": ['online']
                },
                "vocab": "char_vocab", # usually provided by the "token_gather" module
                "max_token_len": 20, # the max length of token, then the output will be max_token_len x token_num (put max_token_len in previor is for padding on token_num)
            },
            "predict": "train",
            "extend_train": "train",
            "online": "train",
        }
    }
    """Config for Token2CharID

    Config Example:
        default_config
    """

    def __init__(self, stage, config: Dict):

        super(Token2CharIDConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.data_pair = self.config.pop('data_pair', {})

        if self.data_set and (not self.data_pair):
            raise ValueError("You must provide 'data_pair' for token2charid.")
        self.vocab = self.config.get('vocab', "")
        if self.data_set and (not self.vocab):
            raise ValueError("You must provide 'vocab' for token2charid.")
        self.max_token_len = self.config['max_token_len']
        self.post_check(self.config, used=[
            "data_pair",
            "data_set",
            "vocab",
            "max_token_len",
        ])


@subprocessor_register('token2charid')
class Token2CharID(ISubProcessor):
    """Use 'Vocabulary' map the character from tokens to id
    """

    def __init__(self, stage: str, config: Token2CharIDConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'token2charid' at stage {self.stage}")
            return
        self.data_pair = config.data_pair


    def process(self, data: Dict)->Dict:
        """Token2CharID Entry

        one_token like 'apple' will generate [1, 2, 2, 3] if max_token_len==4 and the vocab.word2idx = {'a': 1, "p": 2, "l": 3}

        Args:
            data: will process data

        Returns: 
            updated data(token -> char_ids)

        """


        if not self.data_set:
            return data

        def get_index_wrap(sentence_name, offset_name, x):
            """wrap the vocab.get_index"""
            sentence = list(x[sentence_name])
            offsets = x[offset_name]
            char_ids = []
            for offset in offsets:
                token = sentence[offset[0]: offset[1]][:self.config.max_token_len]
                token = token + [vocab.pad] * (self.config.max_token_len-len(token))
                char_ids.append([vocab.get_index(c) for c in token])
            return char_ids

        vocab = Vocabulary.load(data[self.config.vocab])

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do token2charid on it.')
                continue
            data_set = data['data'][data_set_name]
            for key, value in self.data_pair.items():
                sentence_name, offset_name = key.split('&')
                sentence_name = sentence_name.strip()
                offset_name = offset_name.strip()
                get_index = partial(get_index_wrap, sentence_name, offset_name)
                if os.environ.get('DISABLE_PANDAS_PARALLEL', 'false') != 'false':
                    data_set[value] = data_set.parallel_apply(get_index, axis=1)
                else:
                    data_set[value] = data_set.apply(get_index, axis=1)
        return data
