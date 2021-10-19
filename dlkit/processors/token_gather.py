from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import Config
from typing import Dict, Callable
import json
import copy

from dlkit.processors import processor_register, processor_config_register, Processor
from dlkit.processors._util.tokenizer_processor import PreTokenizerFactory, TokenizerPostprocessorFactory, TokenizerNormalizerFactory
from tokenizers import normalizers
from tokenizers import pre_tokenizers

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece


@processor_config_register('token_gather')
class TokenGatherConfig(Config):
    """docstring for GeneralTokenizerConfig
        {
            '_name': 'token_gather'
            '_status': ['train'],
            'config': {
                'data_set': {                   // for different status, this processor will process different part of data
                    'train': ['train', 'dev']
                },
                'gather_columns': ['label'], //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                "deliver": "label_vocab", // output Vocabulary object (the Vocabulary of labels) name. 
                "update": null, // null or another Vocabulary object to update
            },
        }, 
    """

    def __init__(self, parallel, status, **kwargs):
        self.parallel = True # always parallel
        self.data_set = kwargs.pop('data_set', {}).pop(status, [])
        self.gather_columns = kwargs.pop("gather_column", [])
        self.deliver = kwargs.pop("deliver", "")
        if not self.deliver:
            raise ValueError("The 'deliver' value must not be null.")
        self.update = kwargs.pop('update', "")

@processor_register('token_gather')
class TokenGather(Processor):
    """
    """

    def __init__(self, status: str, config: TokenGatherConfig):
        super().__init__(status, config)
        self.status = status
        self.config = config
        self.data_set = config.data_set

    def process(self, data: Dict)->Dict:
        if self.config.update:
            self.vocab = data[self.config.update]
        else:
            self.vocab = Vocabulary(do_strip=True)
        for data_set_name in self.data_set:
            data_set = data['data'][data_set_name]
            data_set = self._process(data_set, self.process_data, self.filed_map)
            data['data'][data_set_name] = data_set
        return data
