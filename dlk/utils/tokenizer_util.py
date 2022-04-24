# Copyright 2021 cstsunfu. All rights reserved.
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

from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, NFD, NFC, StripAccents, Strip
from tokenizers.pre_tokenizers import WhitespaceSplit, ByteLevel, Whitespace, BertPreTokenizer
from tokenizers import Tokenizer
from dlk.utils.logger import Logger

logger = Logger.get_logger()


class TokenizerPostprocessorFactory(object):
    """docstring for TokenizerPostprocessorFactory"""
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def bert(self):
        """bert postprocess

        Returns: 
            bert postprocess

        """
        vocab = self.tokenizer.get_vocab()
        try:
            cls_id = vocab['[CLS]']
            sep_id = vocab['[SEP]']
        except Exception as e:
            # logger.error(f"`[CLS]` or `[SEP]` is not a token in this tokenizer.", )
            logger.exception(f"`[CLS]` or `[SEP]` is not a token in this tokenizer.")
            raise e
            # raise PermissionError(f"`[CLS]` or `[SEP]` is not a token in this tokenizer.")

        def _wrap():
            return TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", cls_id),
                    ("[SEP]", sep_id),
                ],
            )
        return _wrap


    def get(self, name):
        """get postprocess by name 

        Returns: 
            postprocess

        """
        return self.__getattribute__(name)



class PreTokenizerFactory(object):
    """PreTokenizerFactory"""
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def bytelevel(self):
        """byte level pre_tokenizer

        Returns: 
            ByteLevel

        """
        return ByteLevel

    @property
    def bert(self):
        """bert pre_tokenizer

        Returns: 
            BertPreTokenizer

        """
        return BertPreTokenizer

    @property
    def whitespace(self):
        """whitespace pre_tokenizer

        Returns: 
            Whitespace

        """
        return Whitespace

    @property
    def whitespacesplit(self):
        """whitespacesplit pre_tokenizer

        Returns: 
            WhitespaceSplit

        """
        return WhitespaceSplit

    def get(self, name):
        """get pretokenizer by name 

        Returns: 
            postprocess

        """
        return self.__getattribute__(name)


class TokenizerNormalizerFactory(object):
    """TokenizerNormalizerFactory"""
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def lowercase(self):
        """do lowercase normalizers

        Returns: 
            Lowercase

        """
        return Lowercase

    @property
    def nfd(self):
        """do nfd normalizers

        Returns: 
            NFD

        """
        return NFD

    @property
    def nfc(self):
        """do nfc normalizers

        Returns: 
            NFC

        """
        return NFC

    @property
    def strip_accents(self):
        """do strip normalizers

        Returns: 
            StripAccents

        """
        return StripAccents

    @property
    def strip(self):
        """do strip normalizers

        Returns: 
            StripAccents

        """
        return Strip

    def get(self, name):
        """get normalizers by name 

        Returns: 
            Normalizer

        """
        return self.__getattribute__(name)
