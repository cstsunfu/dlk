from dlkit.utils.config import Config
from typing import Dict

from . import processor_register, processor_config_register
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
# from tokenizers.models import BPE
from tokenizers import decoders

tokenizer = Tokenizer.from_file('./wp_bert-wiki.json')
tokenizer.decoder = decoders.WordPiece()
# encode = tokenizer.encode('著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。')

encode = tokenizer.encode('a b c d e a b')

bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

# # bert_tokenizer.pre_tokenizer = Whitespace()
# trainer = WordPieceTrainer(
    # vocab_size=40, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# )
# files = ["./a.md"]
# bert_tokenizer.train(files, trainer)
# # bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

# bert_tokenizer.save("wp_bert-wiki.json", pretty=True)


@processor_config_register('wordpiece_tokenizer')
class WordpieceTokenizerConfig(Config):
    """docstring for GeneralTokenizerConfig"""
    def __init__(self, parallel, status, **kwargs):
        self.data_pair = kwargs.pop('data_pair', {})
        self.data_set = kwargs.pop('data_set', {}).pop(status, [])
        self.parallel = self.parallel
        self.config_path = kwargs.pop('config_path', "")
        self.pretokenizer = kwargs.pop('config_path', "")

        {
            "_name": "wordpiece_tokenizer"
            "_status": ["train", "predict", "online"],
            "config": {
                "data_pair": {
                    "origin": "origin_tokens"
                }, // string or list, to_data[input['data'][data_set[..]]][to_data]=fn(input['data'][data_set[..]][from_data])
                "pre_tokenizer": "whitespace", // if don't set this, will use the default normalizer from config
                "config_path": "./token.json",
                "normalizer": ['NFD', 'Lowercase', 'StripAccents'], // if don't set this, will use the default normalizer from config
                "data_set": {                   // for different status, this processor will process different part of data
                    "train": ['train', 'dev'],
                    "predict": ['predict'],
                    "online": ['online']
                },
            },
        }, //0 the process num
@processor_register('wordpiece_tokenizer')
class WordpieceTokenizer(object):
    """
    """

    def __init__(self, status: str, config: WordpieceTokenizerConfig):
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
