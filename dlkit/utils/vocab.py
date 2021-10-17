from collections import Counter


class Vocabulary(object):
    r"""vocabulary doc:
    """
    
    def __init__(self, do_strip: bool=False, unknown: str=''):
        self.word2idx = {}
        self.idx2word = {}
        self.do_strip = do_strip
        self.word_num = 0
        self.word_count = Counter() # reserved
        self.unknown = unknown
        if unknown:
            self.word_count[unknown] += 1
            self.word2idx[unknown] = self.word_num
            self.idx2word[self.word_num] = unknown
            self.word_num += 1

    def get_index(self, word):
        """get the index of word from this vocab
        """
        if self.do_strip:
            word = word.strip()
        try:
            return self.word2idx[word]
        except:
            if self.unknown:
                return self.word2idx[self.unknown]
            else:
                raise KeyError('Unkown word: {}'.format(word))
            
    def get_word(self, index):
        """get the word of index from this vocab
        """
        try:
            return self.idx2word[index]
        except:
            raise KeyError('Undefined index: {}'.format(index))

    def add(self, word):
        r"""
        """
        if not self.word_count[word]:
            self.word2idx[word] = self.word_num
            self.idx2word[self.word_num] = word
            self.word_num += 1
        self.word_count[word] += 1
        return self
    
    def __len__(self):
        return len(self.word2idx)
    
    def add_from_iter(self, iterator):
        r"""add a list or set of words
        """
        for word in iterator:
            self.add(word)
        return self
