# Copyright cstsunfu. All rights reserved.
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

from dlk.utils.tokenizer_util import PreTokenizerFactory, TokenizerPostprocessorFactory, TokenizerNormalizerFactory
from dlk.utils.config import ConfigTool, BaseConfig
from typing import Dict, Callable, Union
import json
from dlk.utils.logger import Logger

from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from tokenizers import normalizers
from tokenizers import pre_tokenizers

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

logger = Logger.get_logger()


@subprocessor_config_register('fast_tokenizer')
class FastTokenizerConfig(BaseConfig):
    default_config = {
           "_name": "fast_tokenizer",
           "config": {
               "train": {
                   "data_set": {                   # for different stage, this processor will process different part of data
                       "train": ["train", "valid", 'test'],
                       "predict": ["predict"],
                       "online": ["online"]
                   },
                   "config_path": "*@*",
                   "truncation": {     # if this is set to None or empty, will not do trunc
                       "direction": "right", # default `right`, if set `left`, will reserve the rightest chars.
                       "stride": 0, # if the sequence is very long, will split to multiple span, stride is the window slide
                       "max_length": 512,
                       "strategy": "longest_first", # Can be one of longest_first, only_first or only_second.
                   },
                   "normalizer": "default", # ["nfd", "lowercase", "strip_accents", "some_processor_need_config": {config}], # if don't set this, will use the default normalizer from config
                   "pre_tokenizer": "default",# [{"whitespace": {}}], # if don't set this, will use the default normalizer from config
                   "post_processor": "default", # "bert", # if don't set this, will use the default normalizer from config, WARNING: not support disable  the default setting( so the default tokenizer.post_tokenizer should be null and only setting in this configure)
                   "output_map": { # this is the default value, you can provide other name
                       "tokens": "tokens",
                       "ids": "input_ids",
                       "attention_mask": "attention_mask",
                       "type_ids": "type_ids",
                       "special_tokens_mask": "special_tokens_mask",
                       "offsets": "offsets",
                       "word_ids": "word_ids",
                       "sequence_ids": "sequence_ids",
                   }, # the tokenizer output(the key) map to the value
                   "input_map": {
                       "sentence": "sentence", # for sigle input, tokenizer the "sentence"
                       "sentence_a": "sentence_a", # for pair inputs, tokenize the "sentence_a" && "sentence_b"
                       "sentence_b": "sentence_b", # for pair inputs
                       "pretokenized_words": "pretokenized_words", # pretokenized word related to sentence
                       "pretokenized_words_a": "pretokenized_words_a", # pretokenized word b related to sentence_a
                       "pretokenized_words_b": "pretokenized_words_b", # pretokenized word b related to sentence_b
                       "pretokenized_word_offsets": "pretokenized_word_offsets", # pretokenized word offsets for fix offset
                       "pretokenized_word_offsets_a": "pretokenized_word_offsets_a", # pretokenized word offsets for fix offset
                       "pretokenized_word_offsets_b": "pretokenized_word_offsets_b", # pretokenized word offsets for fix offset
                   },
                   "deliver": "tokenizer",
                   "process_data": { "is_pretokenized": False, "add_special_tokens": True},
                   "expand_examples": False, # if the sequence is very long, will split to multiple span, whether expand the examples
                   "data_type": "single", # single or pair, if not provide, will calc by len(process_data)
                   "fix_offset": False, # whether fix the offset for pretokenizerd word
               },
               "predict": ["train", {"deliver": None}],
               "extend_train": ["train", {"deliver": None}],
               "online": ["train", {"deliver": None}],
           }
       }
    """Config for FastTokenizer

    Config Example:
        default_config
    """
    def __init__(self, stage, config):
        super(FastTokenizerConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.config_path = self.config.get('config_path')
        self.normalizer = self.config.get('normalizer', "default")
        self.pre_tokenizer = self.config.get('pre_tokenizer', "default")
        self.post_processor = self.config.get('post_processor', "default")
        self.expand_examples = self.config.get('expand_examples', False)
        self.truncation = self.config["truncation"]
        self.deliver = self.config['deliver']
        self.load = self.config.get('load', None)
        self.input_map = self.config['input_map']
        self.output_map = self.config.get('output_map', { # default
            "tokens": "tokens",
            "ids": "input_ids",
            "attention_mask": "attention_mask",
            "type_ids": "type_ids",
            "special_tokens_mask": "special_tokens_mask",
            "offsets": "offsets",
            "word_ids": "word_ids",
            "sequence_ids": "sequence_ids",
        })
        self.process_data = self.config['process_data']
        self.is_pretokenized = self.process_data['is_pretokenized']
        self.data_type = self.config["data_type"]
        self.fix_offset = self.config['fix_offset']
        assert self.data_type in ['single', 'pair']
        self.post_check(self.config, used=[
            "data_set",
            "config_path",
            "truncation",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "output_map",
            "input_map",
            "fix_offset",
            "deliver",
            "process_data",
            "expand_examples",
            "data_type",
        ])


@subprocessor_register('fast_tokenizer')
class FastTokenizer(ISubProcessor):
    """ FastTokenizer use hugingface tokenizers

    Tokenizer the single $sentence
    Or tokenizer the pair $sentence_a, $sentence_b
    Generator $tokens, $input_ids, $type_ids, $special_tokens_mask, $offsets, $word_ids, $overflowing, $sequence_ids
    """

    def __init__(self, stage: str, config: FastTokenizerConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        if not self.config.data_set:
            logger.info(f"Skip 'fast_tokenizer' at stage {self.stage}")
            return
        self.tokenizer = Tokenizer.from_file(self.config.config_path)
        pretokenizer_factory = PreTokenizerFactory(self.tokenizer)
        tokenizer_postprocessor_factory = TokenizerPostprocessorFactory(self.tokenizer)
        tokenizer_normalizer_factory = TokenizerNormalizerFactory(self.tokenizer)

        if not self.config.pre_tokenizer:
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
        elif self.config.pre_tokenizer != "default":
            assert isinstance(config.pre_tokenizer, list)
            pre_tokenizers_list = []
            for one_pre_tokenizer in config.pre_tokenizer:
                pre_tokenizers_list.append(self._get_processor(pretokenizer_factory, one_pre_tokenizer))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizers_list)

        if not self.config.post_processor:
            raise KeyError("The tokenizer is not support disable default tokenizers post processer. (You can delete the config manully)")
        elif self.config.post_processor != "default":
            self.tokenizer.post_processor = self._get_processor(tokenizer_postprocessor_factory, config.post_processor)

        if not self.config.normalizer:
            self.tokenizer.normalizer = normalizers.Sequence([])
        elif self.config.normalizer != "default":
            assert isinstance(config.normalizer, list)
            normalizers_list = []
            for one_normalizer in config.normalizer:
                normalizers_list.append(self._get_processor(tokenizer_normalizer_factory, one_normalizer))
            self.tokenizer.normalizer = normalizers.Sequence(normalizers_list)

        if self.config.truncation:
            self.tokenizer.enable_truncation(max_length=self.config.truncation['max_length'], 
                                             stride=self.config.truncation["stride"], 
                                             strategy=self.config.truncation['strategy'],
                                             direction=self.config.truncation["direction"])

    def _get_processor(self, factory: Union[PreTokenizerFactory, TokenizerNormalizerFactory, TokenizerPostprocessorFactory], one_processor: Union[Dict, str]):
        """return the processor in factory by the processor name and update the config of the processor if provide

        Args:
            factory: process factory
            one_processor: the processor info, it's name (and config)

        Returns: 
            processor

        """
        if isinstance(one_processor, dict):
            assert len(one_processor) == 1
            process_name, process_config = list(one_processor.items())[0]
            return factory.get(process_name)(**process_config)
        else:
            assert isinstance(one_processor, str)
            return factory.get(one_processor)()

    def _process(self, data: pd.DataFrame)->pd.DataFrame:
        """use self._tokenize tokenize the data

        Args:
            data: several data in dataframe

        Returns: 
            updated dataframe

        """
        if self.config.data_type=='single':
            batch_encodes = self.tokenizer.encode_batch(data[self.config.input_map['pretokenized_words']] if self.config.is_pretokenized else data[self.config.input_map['sentence']], **self.config.process_data)
        else: # pair
            sentence_as = data[self.config.input_map['pretokenized_words_a']] if self.config.is_pretokenized else data[self.config.input_map['sentence_a']]
            sentence_bs = data[self.config.input_map['pretokenized_words_b']] if self.config.is_pretokenized else data[self.config.input_map['sentence_b']]
            batch_encodes = self.tokenizer.encode_batch([(sentence_a, sentence_b) for sentence_a, sentence_b in zip(sentence_as, sentence_bs)], **self.config.process_data)
        if self.config.expand_examples:
            encodes_list = []
            for encode in batch_encodes:
                encodes_list.append([encode]+encode.overflowing)
            data['_tokenizer_encoders'] = encodes_list
            data = data.explode('_tokenizer_encoders', ignore_index=True)
        else:
            data['_tokenizer_encoders'] = batch_encodes
        tokens_list, ids_list, attention_mask_list, type_ids_list, special_tokens_mask_list, offsets_list, word_ids_list, overflowing_list, sequence_ids_list = [], [], [], [], [], [], [], [], []
        for encode in data['_tokenizer_encoders']:
            tokens_list.append(encode.tokens)
            ids_list.append(encode.ids)
            attention_mask_list.append(encode.attention_mask)
            type_ids_list.append(encode.type_ids)
            special_tokens_mask_list.append(encode.special_tokens_mask)
            offsets_list.append(encode.offsets)
            word_ids_list.append(encode.word_ids)
            # overflowing_list.append(encode.overflowing)
            sequence_ids_list.append(encode.sequence_ids)
        output_map = self.config.output_map
        data[output_map['tokens']] = tokens_list
        data[output_map['ids']] = ids_list
        data[output_map['attention_mask']] = attention_mask_list
        data[output_map['type_ids']] = type_ids_list
        data[output_map['special_tokens_mask']] = special_tokens_mask_list
        data[output_map['offsets']] = offsets_list
        data[output_map['word_ids']] = word_ids_list
        # data[output_map['overflowing']] = overflowing_list
        data[output_map['sequence_ids']] = sequence_ids_list
        data.drop("_tokenizer_encoders", axis=1, inplace=True)

        if self.config.is_pretokenized and self.config.fix_offset:
            data[output_map['offsets']] = data.apply(self._fix_offset, axis=1)
        return data

    def _fix_offset(self, one_line: pd.Series):
        """fix the pretokenizerd offset

        Args:
            one_line: a Series which contains the config.input_map['pretokenized_word_offsets'], config.output_map['offsets'], config.output_map['word_ids'], configs.output_map['type_ids']

        Returns: 
            encode.tokens, encode.ids, encode.attention_mask, encode.type_ids, encode.special_tokens_mask, encode.offsets, encode.word_ids, encode.overflowing, encode.sequence_ids 

        """
        fixed_offsets = []
        word_offsets = []
        if self.config.data_type == 'single':
            word_offsets = [one_line[self.config.input_map['pretokenized_word_offsets']]]
        else: # pair
            word_offsets = [one_line[self.config.input_map['pretokenized_word_offsets_a']], one_line[self.config.input_map['pretokenized_word_offsets_b']]]
        for offset, word_id, type_id in zip(one_line['offsets'], one_line['word_ids'], one_line['type_ids']):
            if offset == (0, 0):
                fixed_offsets.append(offset)
            else:
                fixed_offsets.append((offset[0]+word_offsets[type_id][word_id][0], offset[1]+word_offsets[type_id][word_id][0]))
        return fixed_offsets

    def process(self, data: Dict)->Dict:
        """Tokenizer entry

        Args:
            data: 
            >>> {
            >>>     "data": {"train": ...},
            >>>     "tokenizer": ..
            >>> }

        Returns: 
            data and the tokenizer info is in the data['data'], if you set the self.config.deliver, the data[self.config.deliver] will set to self.tokenizer.to_str()

        """
        if not self.config.data_set:
            return data
        for data_set_name in self.config.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip tokenize it.')
                continue
            data_set = data['data'][data_set_name]
            data_set = self._process(data_set)
            data['data'][data_set_name] = data_set
        if self.config.deliver:
            data[self.config.deliver] = self.tokenizer.to_str()
        return data
