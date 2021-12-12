from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import ConfigTool
from typing import Dict, Callable, Set, List
from dlkit.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
import pandas as pd
from dlkit.utils.logger import logger
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit

logger = logger()

@subprocessor_config_register('token_norm')
class TokenNormConfig(object):
    """docstring for TokenNormConfig
        {
            "_name": "token_norm",
            "config": {
                "train":{
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'valid', 'test'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "zero_digits_replaced": true,
                    "lowercase": false,
                    "unk": "[UNK]", //if the unk token is in your vocab, you can provide one to replace it
                    "vocab": "the path to vocab(if the token in vocab skip norm it), the file is setted to one token per line", 
                    "tokenizer": "whitespace_split",  //currently we only support use space split the setentce
                    "data_pair": {
                        "sentence": "norm_sentence"
                    },
                },
                "predict": "train",
                "online": "train",
            }
        }
    """
    def __init__(self, stage, config: Dict):

        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        self.data_pair = self.config.pop('data_pair', {})
        self.zero_digits_replaced = self.config.pop('zero_digits_replaced', True)
        self.lowercase = self.config.pop('lowercase', True)
        tokenizer_name = self.config['tokenizer']
        assert tokenizer_name == 'whitespace_split', f'currently we only support use space split the setentce'
        self.unk = self.config['unk']
        self.vocab = self.get_vocab(self.config.get('vocab', ''))
        self._tokenizer = Tokenizer(WordLevel(vocab={token: i for i, token in enumerate(self.vocab)}, unk_token=self.unk))
        self._tokenizer.pre_tokenizer = WhitespaceSplit()

    def get_vocab(self, path):
        """TODO: Docstring for get_vocab.
        :path: TODO
        :returns: TODO
        """
        vocab = {self.unk}
        if not path:
            return vocab
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                vocab.add(line.strip())
        return vocab

    def tokenizer(self, seq):
        """TODO: Docstring for whitespace_split.
        :returns: TODO
        """
        encode = self._tokenizer.encode(seq)
        return encode


@subprocessor_register('token_norm')
class TokenNorm(ISubProcessor):
    """docstring for TokenNorm
    """

    def __init__(self, stage: str, config: TokenNormConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        self._zero_digits_replaced_num = 0
        self._lower_case_num = 0
        self._lower_case_zero_digits_replaced_num = 0

    def token_norm(self, token: str):
        """norm token, the result len(result) == len(token), exp.  12348->00000
        :token: TODO
        :returns: norm token,
        """
        if self.config.zero_digits_replaced:
            norm = ''
            for c in token:
                if c.isdigit():
                    norm += '0'
                else:
                    norm += c
            if norm in self.config.vocab:
                self._zero_digits_replaced_num += 1
                return norm
        elif self.config.lowercase:
            norm = token.lower()
            if norm in self.config.vocab:
                self._lower_case_num += 1
                return norm
        elif self.config.lowercase and self.config.zero_digits_replaced:
            norm = ''
            for c in token.lower():
                if c.isdigit():
                    norm += '0'
                else:
                    norm += c
            if norm in self.config.vocab:
                self._lower_case_zero_digits_replaced_num += 1
                return norm
        else:
            return None

    def seq_norm(self, seq: str):
        """TODO: Docstring for token_norm.
        :seq: TODO
        :returns: TODO
        """
        norm_seq = [c for c in seq]
        encode = self.config.tokenizer(seq)
        for i, token in enumerate(encode.tokens):
            if token == self.config.unk:
                norm_token = self.token_norm(token)
                if not norm_token:
                    continue
                token_offset = encode.offsets[i]
                assert len(norm_token) == token_offset[1] - token_offset[0]
                norm_seq[token_offset[0]: token_offset[1]] = norm_token
        return ''.join(norm_seq)
        
    def process(self, data: Dict)->Dict:
        '''
            data: {
                "train": list of json format train data
            }
        '''

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do token_norm on it.')
                continue
            data_set = data['data'][data_set_name]

            for key, value in self.config.data_pair.items():
                data_set[value] = data_set.parallel_apply(self.seq_norm, axis=1)
        logger.info(f"We use zero digits to replace digit token num is {self._zero_digits_replaced_num}, do lowercase token num is {self._lower_case_num}, do both num is {self._lower_case_zero_digits_replaced_num}")
        return data
