from dlk.utils.vocab import Vocabulary
from collections import Counter
from typing import Dict, Iterable, List, Union
import pytest

class TestVocabulary(object):
    """generate vocab from tokens(token or Iterable tokens)
       you can dumps the object to dict and load from dict
    """

    @pytest.mark.parametrize('tokens', [
                                 ['b', 'a', 'c'],
                                 ['a', 'c'],
                                 'a'
                             ])
    def test_dumps_and_load(self, tokens):
        """test before and after dump the class is same

        """
        vocab = Vocabulary()
        vocab.auto_update(tokens)
        assert Vocabulary.load(vocab.dumps()).__dict__ == vocab.__dict__

    @pytest.mark.parametrize('tokens', [
                                 ['aa', 'aa', 'ab', 'ba'],
                             ])
    def test_filter_rare(self, tokens):
        """test filter the rare appear words

        """
        vocab = Vocabulary()
        vocab.auto_update(tokens)
        vocab.filter_rare(most_common=1)
        assert vocab.word2idx == {'aa': 0}

        vocab = Vocabulary()
        vocab.auto_update(tokens)
        vocab.filter_rare(min_freq=2)
        assert vocab.word2idx == {'aa': 0}

    @pytest.mark.parametrize('tokens', [
                                 ['aa', 'aa', 'ab', 'ba'],
                             ])
    def test_add_autoadd(self, tokens):
        """test add or autoadd do the same thing
        """
        vocab_a = Vocabulary()
        vocab_a.auto_update(tokens)

        vocab_b = Vocabulary()
        for token in tokens:
            vocab_b.add(token)

        vocab_c = Vocabulary()
        vocab_c.add_from_iter(tokens)
        assert vocab_a.dumps() == vocab_b.dumps() == vocab_c.dumps()
