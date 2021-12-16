from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, NFD, NFC, StripAccents, Strip
from tokenizers.pre_tokenizers import WhitespaceSplit, ByteLevel


class TokenizerPostprocessorFactory(object):
    """docstring for TokenizerPostprocessorFactory"""

    @property
    def bert(self):
        """TODO: Docstring for bert.

        :arg1: TODO
        :returns: TODO

        """
        def _wrap():
            """TODO: Docstring for _wrap.
            :returns: TODO

            """
            return TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", 1),
                    ("[SEP]", 2),
                ],
            )
        return _wrap


    def get(self, name):
        """TODO: Docstring for get.
        """
        return self.__getattribute__(name)



class PreTokenizerFactory(object):
    """docstring for PreTokenizerFactory"""

    @property
    def bytelevel(self):
        """TODO: Docstring for bytelevel.

        :arg1: TODO
        :returns: TODO

        """
        return ByteLevel

    @property
    def whitespace(self):
        """TODO: Docstring for bytelevel.

        :arg1: TODO
        :returns: TODO

        """
        return WhitespaceSplit

    def get(self, name):
        """TODO: Docstring for get.

        :name: TODO
        :returns: TODO

        """
        return self.__getattribute__(name)


class TokenizerNormalizerFactory(object):
    """docstring for TokenizerNormalizerFactory"""

    @property
    def lowercase(self):
        """TODO: Docstring for lowercase.

        :arg1: TODO
        :returns: TODO

        """
        return Lowercase

    @property
    def nfd(self):
        """TODO: Docstring for nfd.

        :arg1: TODO
        :returns: TODO

        """
        return NFD

    @property
    def nfc(self):
        """TODO: Docstring for nfc.
        """
        return NFC

    @property
    def strip_accents(self):
        """
        """
        return StripAccents

    @property
    def strip(self):
        """
        """
        return Strip

    def get(self, name):
        """TODO: Docstring for get.
        """
        return self.__getattribute__(name)
