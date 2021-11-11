from ._util import PreTokenizerFactory, TokenizerPostprocessorFactory, TokenizerNormalizerFactory
from dlkit.utils.config import Config, GetConfigByStageMixin
from typing import Dict, Callable
import json

from dlkit.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from tokenizers import normalizers
from tokenizers import pre_tokenizers

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece


@subprocessor_config_register('wordpiece_tokenizer')
class WordpieceTokenizerConfig(Config, GetConfigByStageMixin):
    """
    docstring for GeneralTokenizerConfig
    {
        "_base": "wordpiece_tokenizer",
        "config": {
            "train": { // you can add some whitespace surround the '&' 
                "data_set": {                   // for different stage, this processor will process different part of data
                    "train": ["train", "dev"],
                    "predict": ["predict"],
                    "online": ["online"]
                },
                "config_path": "*@*",
                "normalizer": ["nfd", "lowercase", "strip_accents", "some_processor_need_config": {config}], // if don't set this, will use the default normalizer from config
                "pre_tokenizer": ["whitespace": {}], // if don't set this, will use the default normalizer from config
                "post_processor": "bert", // if don't set this, will use the default normalizer from config, WARNING: not support disable  the default setting( so the default tokenizer.post_tokenizer should be null and only setting in this configure)
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
                    ["sentence", { "is_pretokenized": false}], 
                ],
                /*"data_type": "pair", // single or pair*/
                /*"process_data": [*/
                    /*['sentence_a', { "is_pretokenized": false}], */ 
                    /*['sentence_b', {}], the config of the second data must as same as the first*/ 
                /*],*/
            },
            "predict": "train",
            "online": "train"
        }
    }
    """
    def __init__(self, stage, config):
        self.config = self.get_config(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        self.config_path = self.config.get('config_path', "")
        self.normalizer = self.config.get('normalizer', "default")
        self.pretokenizer = self.config.get('pre_tokenizer', "default")
        self.post_processor = self.config.get('post_processor', "default")
        self.filed_map = self.config.get('filed_map', { # default
            "tokens": "tokens",
            "ids": "ids",
            "attention_mask": "attention_mask",
            "type_ids": "type_ids",
            "special_tokens_mask": "special_tokens_mask",
            "offsets": "offsets",
        })
        self.process_data = self.config.get("process_data", []) # must provide
        self.data_type = self.config.get("data_type", "single" if len(self.process_data)==1 else "pair" if len(self.process_data)==2 else "UNDEFINED")


@subprocessor_register('wordpiece_tokenizer')
class WordpieceTokenizer(ISubProcessor):
    """
    """

    def __init__(self, stage: str, config: WordpieceTokenizerConfig):
        super().__init__()
        self.stage = stage
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
