from tokenizers.processors import TemplateProcessing


class TokenizerPostprocessorFactory(object):
    """docstring for TokenizerPostprocessorFactory"""
        
    @property
    def bert(self):
        """TODO: Docstring for bert.

        :arg1: TODO
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


    def get(self, name):
        """TODO: Docstring for get.
        """
        return self.__getattribute__(name)
