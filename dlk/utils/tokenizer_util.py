# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging

from tokenizers import Tokenizer
from tokenizers.normalizers import NFC, NFD, Lowercase, Strip, StripAccents
from tokenizers.pre_tokenizers import (
    BertPreTokenizer,
    ByteLevel,
    Whitespace,
    WhitespaceSplit,
)
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)


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
            cls_id = vocab["[CLS]"]
            sep_id = vocab["[SEP]"]
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
