from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace, ByteLevel


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
        return Whitespace

    def get(self, name):
        """TODO: Docstring for get.

        :name: TODO
        :returns: TODO

        """
        return self.__getattribute__(name)
