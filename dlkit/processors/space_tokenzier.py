from dlkit.utils.config import Config
from typing import Dict

from . import processor_register, processor_config_register
import pandas as pd


@processor_config_register('basic')
class SpaceTokenizerConfig(Config):
    """docstring for BasicConfig"""
    def __init__(self, parallel, status, **kwargs):
        self.data_pair = kwargs.pop('data_pair', {})
        self.char_token_idx_map = kwargs.pop('char_token_idx_map', "")
        self.data_set = kwargs.pop('data_set', {}).pop(status, [])
        self.parallel = self.parallel

@processor_register('space_tokenizer')
class SpaceTokenizer(object):
    """
    """

    def __init__(self, status: str, config: SpaceTokenizerConfig):
        super().__init__()
        self.config = config
        self.status = status

    @classmethod
    def tokenize(cls, inp: pd.Series, name: str):
        """TODO: Docstring for tokenize.

        :arg1: TODO
        :returns: TODO
        """
        return inp[name].split(' ')

    def process(self, data: Dict)->Dict:
        for part in self.config.data_set:
            for source, to in self.config.data_pair.items():
                source = source.split('&')
                to = to.split('&')
                assert len(source) == 1
                assert len(to) == 1
                if self.config.parallel:
                    data[part][to[0]] = data[part][source].parallel_apply(SpaceTokenizer.tokenize, axis=1, args=source)
                else:
                    data[part][to[0]] = data[part][source].apply(SpaceTokenizer.tokenize, axis=1, args=source)

        return data
            # "_name": "space_tokenizer"
            # "_status": ["train", "predict", "online"],
            # "config": {
                # "data_pair": {
                    # "origin": "origin_tokens"
                # }, // string or list, to_data[input['data'][data_set[..]]][to_data]=fn(input['data'][data_set[..]][from_data])
                # "map_char_token_idx": "origin_char_token_idx_map", // if this is empty string, will not do this
                # "data_set": {                   // for different status, this processor will process different part of data
                    # "train": ['train', 'dev'],
                    # "predict": ['predict'],
                    # "online": ['online']
                # },
            # },
