from dlkit.utils.config import Config
from typing import Dict, Callable
import json

from dlkit.processors import processor_register, processor_config_register, Processor
from dlkit.processors._util.tokenizer_processor import PreTokenizerFactory, TokenizerPostprocessorFactory, TokenizerNormalizerFactory
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
                'data_set': {                   // for different status, this processor will process different part of data
                    'train': ['train', 'dev'],
                    'predict': ['predict'],
                    'online': ['online']
                },
                'config_path': './token.json',
                "normalizer": ['nfd', 'lowercase', 'strip_accents', "some_processor_need_config": {config}], // if don't set this, will use the default normalizer from config
                "pre_tokenizer": ["whitespace": {}], // if don't set this, will use the default normalizer from config
                'post_processor': 'bert', // if don't set this, will use the default normalizer from config, WARNING: not support disable  the default setting( so the default tokenizer.post_tokenizer should be null and setting in this configure)
                "filed_map": { // this is the default value, you can provide other name
                    "tokens": "tokens",
                    "ids": "ids",
                    "attention_mask": "attention_mask",
                    "type_ids": "type_ids",
                    "special_tokens_mask": "special_tokens_mask",
                    "offsets": "offsets",
                }, // the tokenizer output(the key) map to the value
                "data_type": "single", // single or pair, if not provide, will calc by len(process_data)
                "process_data": [
                    ['sentence', { "is_pretokenized": false}], 
                ],
                /*"data_type": "pair", // single or pair*/
                /*"process_data": [*/
                    /*['sentence_a', { "is_pretokenized": false}], */ 
                    /*['sentence_b', {}], the config of the second data must as same as the first*/ 
                /*],*/
            },
        }, 
    """
    def __init__(self,  status, **kwargs):
        self.data_set = kwargs.pop('data_set', {}).pop(status, [])
        self.config_path = kwargs.pop('config_path', "")
        self.normalizer = kwargs.pop('normalizer', "default")
        self.pretokenizer = kwargs.pop('pre_tokenizer', "default")
        self.post_processor = kwargs.pop('post_processor', "default")
        self.filed_map = kwargs.pop('filed_map', { # default
            "tokens": "tokens",
            "ids": "ids",
            "attention_mask": "attention_mask",
            "type_ids": "type_ids",
            "special_tokens_mask": "special_tokens_mask",
            "offsets": "offsets",
        })
        self.process_data = kwargs.pop("process_data") # must provide
        self.data_type = kwargs.pop("data_type", "single" if len(self.process_data)==1 else "pair" if len(self.process_data)==2 else "UNDEFINED")


@processor_register('wordpiece_tokenizer')
class WordpieceTokenizer(Processor):
    """
    """

    def __init__(self, status: str, config: WordpieceTokenizerConfig):
        super().__init__(status, config)
        self.status = status
        self.tokenizer = Tokenizer.from_file(config.config_path)
        pretokenizer_factory = PreTokenizerFactory()
        tokenizer_postprocessor_factory = TokenizerPostprocessorFactory()
        tokenizer_normalizer_factory = TokenizerNormalizerFactory()
        self.filed_map = config.filed_map
        self.data_set = config.data_set
        self.process_data = config.process_data
        self.data_type = config.data_type

        if self.data_type=='single':
            assert len(self.process_data) == 1
            self._process = self._single
        elif self.data_type == 'pair':
            assert len(self.process_data) == 2
            self._process = self._pair
        else:
            raise KeyError('We only support single or pair data now.')

        if not config.pretokenizer:
            self.tokenizer.pretokenizer = pre_tokenizers.Sequence([])
        elif config.pretokenizer != "default":
            assert isinstance(config.pretokenizer, list)
            pretokenizers_list = []
            for one_pretokenizer in config.pretokenizer:
                pretokenizers_list.append(self._get_processor(pretokenizer_factory, one_pretokenizer))
            self.tokenizer.pretokenizer = pre_tokenizers.Sequence(pretokenizers_list)

        if not config.post_processor:
            raise KeyError("The tokenizer is not support disable default tokenizers post processer. (You can delete the config manully)")
        elif config.post_processor != "default":
            self.tokenizer.post_processor = self._get_processor(tokenizer_postprocessor_factory, config.post_processor)

        if config.normalizer is None:
            self.tokenizer.normalizer = normalizers.Sequence([])
        elif config.normalizer != "default":
            assert isinstance(config.normalizer, list)
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
            return factory.get(process_name)(**process_config)
        else:
            assert isinstance(one_processor, str)
            return factory.get(one_processor)()
    
    def _extend_encoded_token(self, all_tokens, filed_map):
        """TODO: Docstring for _extend_encoded_token.

        :all_tokens: TODO
        :filed_map: TODO
        :returns: TODO

        """
        targets_map = {}
        for k in filed_map:
            targets_map[filed_map[k]] = []
        for token in all_tokens:
            for k in filed_map:
                targets_map[filed_map[k]].append(getattr(token, k))
        return targets_map

    def _single(self, data, process_data, filed_map):
        """TODO: Docstring for _single.

        :data: TODO
        :returns: TODO

        """
        assert len(process_data) == 1
        process_column_name, config = process_data[0]
        process_column_data = data[process_column_name]
        print(self.tokenizer.post_processor)
        all_token = self.tokenizer.encode_batch(process_column_data, **config)
        token_filed_name_value_map = self._extend_encoded_token(all_token, filed_map)
        for k in token_filed_name_value_map:
            data[k] = token_filed_name_value_map[k]
        return data

    def _pair(self, data, process_data, filed_map):
        """TODO: Docstring for _pair.

        :data: TODO
        :returns: TODO

        """
        assert len(process_data) == 2
        process_column_name, config = process_data[0]
        process_column_name_b, _ = process_data[1]
        process_column_data = list(data[[process_column_name, process_column_name_b]].values)
        all_token = self.tokenizer.encode_batch(process_column_data, **config)
        token_filed_name_value_map = self._extend_encoded_token(all_token, filed_map)
        for k in token_filed_name_value_map:
            data[k] = token_filed_name_value_map[k]
        return data

    def process(self, data: Dict)->Dict:
        for data_set_name in self.data_set:
            data_set = data['data'][data_set_name]
            data_set = self._process(data_set, self.process_data, self.filed_map)
            data['data'][data_set_name] = data_set
        return data
