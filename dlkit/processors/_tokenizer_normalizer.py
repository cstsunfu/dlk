from tokenizers.normalizers import Lowercase, NFD, NFC, StripAccents, Strip


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

    def get(self, name):
        """TODO: Docstring for get.
        """
        return self.__getattribute__(name)
