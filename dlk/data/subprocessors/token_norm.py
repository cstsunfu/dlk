"""
This part could merged to fast_tokenizer(it will save some time), but not all process need this part(except some special dataset like conll2003), and will make the fast_tokenizer be heavy.
Token norm:
    Love -> love
    3281 -> 0000
"""
from logging import PercentStyle
from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
import pandas as pd
from dlk.utils.logger import Logger
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit

logger = Logger.get_logger()

@subprocessor_config_register('token_norm')
class TokenNormConfig(BaseConfig):
    """docstring for TokenNormConfig
        {
            "_name": "token_norm",
            "config": {
                "train":{
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'valid', 'test', 'predict'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "zero_digits_replaced": true,
                    "lowercase": true,
                    "extend_vocab": "", //when lowercase is true, this upper_case_vocab will collection all tokens the token is not in vocab but it's lowercase is in vocab. this is only for token gather process
                    "tokenizer": "whitespace_split",  //the path to vocab(if the token in vocab skip norm it), the file is setted to one token per line
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

        super(TokenNormConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.data_pair = self.config.pop('data_pair', {})
        self.zero_digits_replaced = self.config.pop('zero_digits_replaced', True)
        self.lowercase = self.config.pop('lowercase', True)
        self.tokenizer = Tokenizer.from_file(self.config['tokenizer'])
        self.unk = self.tokenizer.model.unk_token
        self.vocab = self.tokenizer.get_vocab()
        self.do_extend_vocab = self.config['extend_vocab']
        self.prefix = self.tokenizer.model.continuing_subword_prefix
        self.post_check(self.config, used=[
            "data_set",
            "zero_digits_replaced",
            "lowercase",
            "extend_vocab",
            "tokenizer",
            "data_pair",
        ])

    def tokenize(self, seq):
        """TODO: Docstring for whitespace_split.
        :returns: TODO
        """
        encode = self.tokenizer.encode(seq)
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
        if not self.data_set:
            logger.info(f"Skip 'token_norm' at stage {self.stage}")
            return
        if self.config.do_extend_vocab:
            self.extend_vocab = set()
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
            digit_num = 0
            for c in token:
                if c.isdigit() or c=='.':
                    norm += '0'
                    digit_num += 1
                else:
                    norm += c
            if norm in self.config.vocab or self.config.prefix+norm in self.config.vocab:
                self._zero_digits_replaced_num += 1
                return norm
            elif self.config.do_extend_vocab and digit_num == len(token) and digit_num<20:
                self.extend_vocab.add(norm)

        if self.config.lowercase:
            norm = token.lower()
            if norm in self.config.vocab or self.config.prefix+norm in self.config.vocab:
                self._lower_case_num += 1
                if self.config.do_extend_vocab:
                    self.extend_vocab.add(token)
                else:
                    raise PermissionError
                return norm

        if self.config.lowercase and self.config.zero_digits_replaced:
            norm = ''
            for c in token.lower():
                if c.isdigit() or c=='.':
                    norm += '0'
                else:
                    norm += c
            if norm in self.config.vocab or self.config.prefix+norm in self.config.vocab:
                self._lower_case_zero_digits_replaced_num += 1
                return norm
        return None

    def seq_norm(self, key:str, one_item: pd.Series):
        """TODO: Docstring for token_norm.
        :seq: TODO
        :returns: TODO
        """
        seq = one_item[key]
        norm_seq = [c for c in seq]
        encode = self.config.tokenize(seq)
        for i, token in enumerate(encode.tokens):
            if token == self.config.unk:
                token_offset = encode.offsets[i]
                prenorm_token = seq[token_offset[0]: token_offset[1]]
                norm_token = self.token_norm(prenorm_token)
                if not norm_token:
                    continue
                assert len(norm_token) == token_offset[1] - token_offset[0], f"Prenorm '{prenorm_token}', postnorm: '{norm_token}' and {len(norm_token)}!= {token_offset[1]} - {token_offset[0]}"
                norm_seq[token_offset[0]: token_offset[1]] = norm_token
        return ''.join(norm_seq)

    def process(self, data: Dict)->Dict:
        '''
        '''

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do token_norm on it.')
                continue
            data_set = data['data'][data_set_name]

            for key, value in self.config.data_pair.items():
                _seq_norm = partial(self.seq_norm, key)
                data_set[value] = data_set.apply(_seq_norm, axis=1)
                # WARNING: if you change the apply to parallel_apply, you should change the _zero_digits_replaced_num, etc. to multiprocess safely(BTW, use parallel_apply in tokenizers==0.10.3 will make the process very slow)
                # data_set[value] = data_set.apply(_seq_norm, axis=1)
        logger.info(f"We use zero digits to replace digit token num is {self._zero_digits_replaced_num}, do lowercase token num is {self._lower_case_num}, do both num is {self._lower_case_zero_digits_replaced_num}")
        if self.config.do_extend_vocab:
            logger.info(f"We will extend {len(self.extend_vocab)} tokens and deliver to {self.config.do_extend_vocab}")
            data[self.config.do_extend_vocab] = list(self.extend_vocab)
        return data
