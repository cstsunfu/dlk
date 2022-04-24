import pandas as pd
import pytest
import os
import copy
from dlk.data.subprocessors.fast_tokenizer import FastTokenizer, FastTokenizerConfig
from dlk.utils.get_root import get_root
import json


@pytest.fixture
def default_single_config(request):
    return {
        "_name": "fast_tokenizer",
        "config": {
            "train": {
                "data_set": {
                    "train": ["train", "valid", 'test', 'predict'],
                    "predict": ["predict"],
                    "online": ["online"]
                },
                "config_path": "",
                "truncation": {
                    "max_length": 10,
                    "strategy": "longest_first"
                },
                "normalizer": 'default',
                "pre_tokenizer": "default",
                "post_processor": "default",
                "output_map": {
                    "tokens": "tokens",
                    "ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "type_ids": "type_ids",
                    "special_tokens_mask": "special_tokens_mask",
                    "offsets": "offsets",
                    "word_ids": "word_ids",
                    "overflowing": "overflowing",
                    "sequence_ids": "sequence_ids",
                },
                "input_map": {
                    "sentence": "sentence",
                },
                "deliver": "tokenizer",
                "process_data": { "is_pretokenized": False},
                "data_type": "single",
            },
            "predict": ["train", {"deliver": None}],
            "online": ["train", {"deliver": None}],
        }
    }


class TestFastTokenizer(object):
    def test_default_tokenizer(self, default_single_config):
        default_single_config = copy.deepcopy(default_single_config)
        tokenizer_path = os.path.join(get_root(), 'tests/data/tokenizer/vocab_tokenizer.json')
        default_single_config['config']['train']['config_path'] = tokenizer_path
        tokenizer_config = FastTokenizerConfig(stage='train', config=default_single_config)
        tokenizer = FastTokenizer(stage='train', config=tokenizer_config)
        data = {
            "data": {
                "train": pd.DataFrame(data={'sentence': ["I have an apple."]})
            }
        }
        result = tokenizer.process(data)
        result = result['data']['train'].iloc[0]
        assert result['sentence'] == "I have an apple."
        assert result['tokens'] == ['I', 'have', 'an', 'app', '##le', '.']
        assert result['input_ids'] == [8, 9, 10, 6, 7, 11]
        assert result['attention_mask'] == [1, 1, 1, 1, 1, 1]
        assert result['type_ids'] == [0, 0, 0, 0, 0, 0]
        assert result['special_tokens_mask'] == [0, 0, 0, 0, 0, 0]
        assert result['offsets'] == [(0, 1), (2, 6), (7, 9), (10, 13), (13, 15), (15, 16)]
        assert result['word_ids'] == [0, 1, 2, 3, 3, 4]
        assert result['sequence_ids'] == [0, 0, 0, 0, 0, 0]

    def test_post_bert_prosess_tokenizer(self, default_single_config):
        default_single_config = copy.deepcopy(default_single_config)
        tokenizer_path = os.path.join(get_root(), 'tests/data/tokenizer/vocab_tokenizer.json')
        default_single_config['config']['train']['config_path'] = tokenizer_path
        default_single_config['config']['train']['post_processor'] = 'bert'
        tokenizer_config = FastTokenizerConfig(stage='train', config=default_single_config)
        tokenizer = FastTokenizer(stage='train', config=tokenizer_config)
        data = {
            "data": {
                "train": pd.DataFrame(data={'sentence': ["I have an apple."]})
            }
        }
        result = tokenizer.process(data)
        result = result['data']['train'].iloc[0]
        assert result['sentence'] == "I have an apple."
        assert result['tokens'] == ['[CLS]', 'I', 'have', 'an', 'app', '##le', '.', '[SEP]']
        assert result['input_ids'] == [0, 8, 9, 10, 6, 7, 11, 1]
        assert result['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1]
        assert result['type_ids'] == [0, 0, 0, 0, 0, 0, 0, 0]
        assert result['special_tokens_mask'] == [1, 0, 0, 0, 0, 0, 0, 1]
        assert result['offsets'] == [(0, 0), (0, 1), (2, 6), (7, 9), (10, 13), (13, 15), (15, 16), (0, 0)]
        assert result['word_ids'] == [None, 0, 1, 2, 3, 3, 4, None]
        assert result['sequence_ids'] == [None, 0, 0, 0, 0, 0, 0, None]

    def test_pre_tokenized_tokenizer(self, default_single_config):
        default_single_config = copy.deepcopy(default_single_config)
        tokenizer_path = os.path.join(get_root(), 'tests/data/tokenizer/vocab_tokenizer.json')
        default_single_config['config']['train']['config_path'] = tokenizer_path
        default_single_config['config']['train']['process_data']['is_pretokenized'] = True
        # default_single_config['config']['train']['post_processor'] = 'bert'
        tokenizer_config = FastTokenizerConfig(stage='train', config=default_single_config)
        tokenizer = FastTokenizer(stage='train', config=tokenizer_config)
        data = {
            "data": {
                "train": pd.DataFrame(data={'sentence': [["我", "来自", '山东', '济宁', '.']]})
            }
        }
        result = tokenizer.process(data)
        result = result['data']['train'].iloc[0]
        assert result['sentence'] == ["我", "来自", '山东', '济宁', '.']
        assert result['tokens'] == ['我', '[UNK]', '山', '##东', '济宁', '.']
        assert result['input_ids'] == [12, 4, 15, 17, 18, 11]
        assert result['attention_mask'] == [1, 1, 1, 1, 1, 1]
        assert result['type_ids'] == [0, 0, 0, 0, 0, 0]
        assert result['special_tokens_mask'] == [0, 0, 0, 0, 0, 0]
        assert result['offsets'] == [(0, 1), (0, 2), (0, 1), (1, 2), (0, 2), (0, 1)]
        assert result['word_ids'] == [0, 1, 2, 2, 3, 4]
        assert result['sequence_ids'] == [0, 0, 0, 0, 0, 0]

    @pytest.mark.skip("Not implement yet")
    def test_pre_tokenized_rel_offset_tokenizer(self, default_single_config):
        default_single_config = copy.deepcopy(default_single_config)
        tokenizer_path = os.path.join(get_root(), 'tests/data/tokenizer/vocab_tokenizer.json')
        default_single_config['config']['train']['config_path'] = tokenizer_path
        default_single_config['config']['train']['process_data']['is_pretokenized'] = True
        default_single_config['config']['train']['pre_offsets'] = True # TODO: currently is not support this
        tokenizer_config = FastTokenizerConfig(stage='train', config=default_single_config)
        tokenizer = FastTokenizer(stage='train', config=tokenizer_config)
        data = {
            "data": {
                "train": pd.DataFrame(data={
                    'sentence': [["我", "来自", '山东', '济宁', '.']], 
                    "origin_sentence": ['我来自山东济宁.'], 
                    "origin_token_offsets": [
                        (0, 1), (1, 3), (3, 5), (5, 7), (7, 8)
                    ]
                })
            }
        }
        result = tokenizer.process(data)
        result = result['data']['train'].iloc[0]
        assert result['origin_sentence'] == "我来自山东济宁."
        assert result['sentence'] == ["我", "来自", '山东', '济宁', '.']
        assert result['tokens'] == ['我', '[UNK]', '山', '##东', '济宁', '.']
        assert result['input_ids'] == [12, 4, 15, 17, 18, 11]
        assert result['attention_mask'] == [1, 1, 1, 1, 1, 1]
        assert result['type_ids'] == [0, 0, 0, 0, 0, 0]
        assert result['special_tokens_mask'] == [0, 0, 0, 0, 0, 0]
        assert result['offsets'] == [(0, 1), (1, 3), (3, 4), (4, 5), (5, 7), (7, 8)] # just as offsets + origin_token_offsets[word_ids[i]][0]
        assert result['word_ids'] == [0, 1, 2, 2, 3, 4]
        assert result['sequence_ids'] == [0, 0, 0, 0, 0, 0]

    def test_pair_tokenizer(self, default_single_config):
        default_pair_config = copy.deepcopy(default_single_config)
        tokenizer_path = os.path.join(get_root(), 'tests/data/tokenizer/vocab_tokenizer.json')
        default_pair_config['config']['train']['config_path'] = tokenizer_path
        default_pair_config['config']['train']['data_type'] = 'pair'
        default_pair_config['config']['train']['input_map'] = {
            "sentence_a": "sentence_a",
            "sentence_b": "sentence_b",
        }
        default_pair_config['config']['train']['truncation'] = {
            "max_length": 20,
            "strategy": "longest_first"
        }

        tokenizer_config = FastTokenizerConfig(stage='train', config=default_pair_config)
        tokenizer = FastTokenizer(stage='train', config=tokenizer_config)
        data = {
            "data": {
                "train": pd.DataFrame(data={'sentence_a': ["I have an apple."], 'sentence_b': ['an apple.']})
            }
        }
        result = tokenizer.process(data)
        result = result['data']['train'].iloc[0]
        assert result['sentence_a'] == "I have an apple."
        assert result['sentence_b'] == "an apple."
        assert result['tokens'] == ['I', 'have', 'an', 'app', '##le', '.', 'an', 'app', '##le', '.']
        assert result['input_ids'] == [8, 9, 10, 6, 7, 11, 10, 6, 7, 11]
        assert result['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert result['type_ids'] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        assert result['special_tokens_mask'] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert result['offsets'] == [(0, 1), (2, 6), (7, 9), (10, 13), (13, 15), (15, 16), (0, 2), (3, 6), (6, 8), (8, 9)]
        assert result['word_ids'] == [0, 1, 2, 3, 3, 4, 0, 1, 1, 2] 
        assert result['sequence_ids'] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

    def test_post_bert_pair_tokenizer(self, default_single_config):
        default_pair_config = copy.deepcopy(default_single_config)
        tokenizer_path = os.path.join(get_root(), 'tests/data/tokenizer/vocab_tokenizer.json')
        default_pair_config['config']['train']['config_path'] = tokenizer_path
        default_pair_config['config']['train']['data_type'] = 'pair'
        default_pair_config['config']['train']['post_processor'] = 'bert'
        default_pair_config['config']['train']['input_map'] = {
            "sentence_a": "sentence_a",
            "sentence_b": "sentence_b",
        }
        default_pair_config['config']['train']['truncation'] = {
            "max_length": 20,
            "strategy": "longest_first"
        }

        tokenizer_config = FastTokenizerConfig(stage='train', config=default_pair_config)
        tokenizer = FastTokenizer(stage='train', config=tokenizer_config)
        data = {
            "data": {
                "train": pd.DataFrame(data={'sentence_a': ["I have an apple."], 'sentence_b': ['an apple.']})
            }
        }
        result = tokenizer.process(data)
        result = result['data']['train'].iloc[0]
        assert result['sentence_a'] == "I have an apple."
        assert result['sentence_b'] == "an apple."
        assert result['tokens'] == ['[CLS]', 'I', 'have', 'an', 'app', '##le', '.', '[SEP]', 'an', 'app', '##le', '.', '[SEP]']
        assert result['input_ids'] == [0, 8, 9, 10, 6, 7, 11, 1, 10, 6, 7, 11, 1]
        assert result['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert result['type_ids'] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert result['special_tokens_mask'] == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        assert result['offsets'] == [(0, 0), (0, 1), (2, 6), (7, 9), (10, 13), (13, 15), (15, 16), (0, 0), (0, 2), (3, 6), (6, 8), (8, 9), (0, 0)]
        assert result['word_ids'] == [None, 0, 1, 2, 3, 3, 4, None, 0, 1, 1, 2, None] 
        assert result['sequence_ids'] == [None, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, None]

    def test_post_bert_pair_trunc_tokenizer(self, default_single_config):
        default_pair_config = copy.deepcopy(default_single_config)
        tokenizer_path = os.path.join(get_root(), 'tests/data/tokenizer/vocab_tokenizer.json')
        default_pair_config['config']['train']['config_path'] = tokenizer_path
        default_pair_config['config']['train']['data_type'] = 'pair'
        default_pair_config['config']['train']['post_processor'] = 'bert'
        default_pair_config['config']['train']['input_map'] = {
            "sentence_a": "sentence_a",
            "sentence_b": "sentence_b",
        }
        default_pair_config['config']['train']['truncation'] = {
            "max_length": 11,
            "strategy": "longest_first"
        }

        tokenizer_config = FastTokenizerConfig(stage='train', config=default_pair_config)
        tokenizer = FastTokenizer(stage='train', config=tokenizer_config)
        data = {
            "data": {
                "train": pd.DataFrame(data={'sentence_a': ["I have an apple."], 'sentence_b': ['an apple.']})
            }
        }
        result = tokenizer.process(data)
        result = result['data']['train'].iloc[0]
        # print(result)
        assert result['sentence_a'] == "I have an apple."
        assert result['sentence_b'] == "an apple."
        assert result['tokens'] == ['[CLS]', 'I', 'have', 'an', 'app', '[SEP]', 'an', 'app', '##le', '.', '[SEP]']
        assert result['input_ids'] == [0, 8, 9, 10, 6, 1, 10, 6, 7, 11, 1]
        assert result['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert result['type_ids'] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert result['special_tokens_mask'] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        assert result['offsets'] == [(0, 0), (0, 1), (2, 6), (7, 9), (10, 13), (0, 0), (0, 2), (3, 6), (6, 8), (8, 9), (0, 0)]
        assert result['word_ids'] == [None, 0, 1, 2, 3, None, 0, 1, 1, 2, None] 
        assert result['sequence_ids'] == [None, 0, 0, 0, 0, None, 1, 1, 1, 1, None]
