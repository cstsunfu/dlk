"""
Use 'Vocabulary' map the tokens to id
"""
from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
from dlk.utils.logger import Logger

logger = Logger.get_logger()

@subprocessor_config_register('token2id')
class Token2IDConfig(BaseConfig):
    """docstring for Token2IDConfig
        {
            "_name": "token2id",
            "config": {
                "train":{ //train、predict、online stage config,  using '&' split all stages
                    "data_pair": {
                        "labels": "label_ids"
                    },
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'valid', 'test', 'predict'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "vocab": "label_vocab", // usually provided by the "token_gather" module
                }, //3
                "predict": "train",
                "online": "train",
            }
        }
    """

    def __init__(self, stage, config: Dict):

        super(Token2IDConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.data_pair = self.config.pop('data_pair', {})
        if self.data_set and (not self.data_pair):
            raise ValueError("You must provide 'data_pair' for token2id.")
        self.vocab = self.config.get('vocab', "")
        if self.data_set and (not self.vocab):
            raise ValueError("You must provide 'vocab' for token2id.")
        self.post_check(self.config, used=[
            "data_pair",
            "data_set",
            "vocab",
            "max_token_len",
        ])


@subprocessor_register('token2id')
class Token2ID(ISubProcessor):
    """docstring for Token2ID
    """

    def __init__(self, stage: str, config: Token2IDConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'token2id' at stage {self.stage}")
            return
        self.data_pair = config.data_pair


    def process(self, data: Dict)->Dict:

        if not self.data_set:
            return data

        vocab = Vocabulary.load(data[self.config.vocab])

        def get_index_wrap(key, x):
            """TODO: Docstring for get_index_wrap.
            """
            return vocab.auto_get_index(x[key])

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do token2id on it.')
                continue
            data_set = data['data'][data_set_name]
            for key, value in self.data_pair.items():
                get_index = partial(get_index_wrap, key)
                data_set[value] = data_set.parallel_apply(get_index, axis=1)
        return data
