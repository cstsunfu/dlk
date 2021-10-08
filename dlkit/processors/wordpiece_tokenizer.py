from dlkit.utils.config import Config
from typing import Dict, Callable
import json

from . import processor_register, processor_config_register
from _pretokenizer import PreTokenizerFactory
from _tokenizer_postprocesser import TokenizerPostprocessorFactory
from _tokenizer_normalizer import TokenizerNormalizerFactory
from tokenizers import normalizers
from tokenizers import pre_tokenizers

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece


@processor_config_register('wordpiece_tokenizer')
class WordpieceTokenizerConfig(Config):
    """docstring for GeneralTokenizerConfig
        {
            '_name': 'wordpiece_tokenizer'
            '_status': ['train', 'predict', 'online'],
            'config': {
                'data_pair': {
                    'origin': 'origin_tokens'
                }, // string or list, to_data[input['data'][data_set[..]]][to_data]=fn(input['data'][data_set[..]][from_data])
                'pre_tokenizer': 'whitespace', // if don't set this, will use the default normalizer from config, set to null, "" or [] will disable the default setting
                'post_tokenizer': 'bert', // if don't set this, will use the default normalizer from config, WARNING: not support disable  the default setting( so the default tokenizer.post_tokenizer should be null and setting in this configure)
                'config_path': './token.json',
                'normalizer': ['NFD', 'Lowercase', 'StripAccents', "NeedConfig": {config}], // if don't set this, will use the default normalizer from config 
                'data_set': {                   // for different status, this processor will process different part of data
                    'train': ['train', 'dev'],
                    'predict': ['predict'],
                    'online': ['online']
                },
            },
        }, //0 the process num

    """
    def __init__(self, parallel, status, **kwargs):
        self.parallel = parallel
        self.data_set = kwargs.pop('data_set', {}).pop(status, [])
        self.data_pair = kwargs.pop('data_pair', {})
        self.config_path = kwargs.pop('config_path', "")
        self.pretokenizer = kwargs.pop('pre_tokenizer', "default")
        self.normalizer = kwargs.pop('normalizer', "default")
        self.post_processor = kwargs.pop('post_processor', "default")

@processor_register('wordpiece_tokenizer')
class WordpieceTokenizer(object):
    """
    """

    def __init__(self, status: str, config: WordpieceTokenizerConfig):
        super().__init__()
        self.status = status
        self.tokenizer = Tokenizer.from_file(config.config_path)
        pretokenizer_factory = PreTokenizerFactory()
        tokenizer_postprocessor_factory = TokenizerPostprocessorFactory()
        tokenizer_normalizer_factory = TokenizerNormalizerFactory()

        if not config.pretokenizer:
            self.tokenizer.pretokenizer = pre_tokenizers.Sequence([])
        elif config.pretokenizer != "default":
            assert isinstance(config.post_processor, list)
            pretokenizers_list = []
            for one_pretokenizer in config.pretokenizer:
                pretokenizers_list.append(self._get_processor(pretokenizer_factory, one_pretokenizer))
            self.tokenizer.pretokenizer = pre_tokenizers.Sequence(pretokenizers_list)

        if not config.post_processor:
            raise KeyError("The tokenizer is not support disable default tokenizers post processer. (You can delete the config manully)")
        elif config.post_processor != "default":
            self.tokenizer.posttokenizer = self._get_processor(tokenizer_postprocessor_factory, config.post_processor)

        if config.normalizer is None:
            self.tokenizer.normalizer = normalizers.Sequence([])
        elif config.normalizer != "default":
            assert isinstance(config.post_processor, list)
            normalizers_list = []
            for one_normalizer in config.normalizer:
                normalizers_list.append(self._get_processor(tokenizer_normalizer_factory, one_normalizer))
            self.tokenizer.normalizer = normalizers.Sequence(normalizers_list)

    def _get_processor(self, factory, one_processor):
        """TODO: Docstring for _get_processor.

        :factory: TODO
        :one_processor: TODO
        :returns: TODO

        """
        if isinstance(one_processor, dict):
            assert len(one_processor) == 1
            process_name, process_config = list(one_processor.items())[0]
            return factory.get(process_name)(process_config)
        else:
            assert isinstance(one_processor, str)
            return factory.get(one_processor)()

    def process(self, data: Dict)->Dict:
        return data
