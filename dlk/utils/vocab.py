# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import Counter
from typing import Dict, Iterable, List, Union

import pandas as pd

from dlk.utils.io import open


class Vocabulary(object):
    """generate vocab from tokens(token or Iterable tokens)
    you can dumps the object to dict and load from dict
    """

    def __init__(
        self, do_strip: bool = False, unknown: str = "", ignore: str = "", pad: str = ""
    ):
        self.word2idx = {}
        self.idx2word = {}
        self.do_strip = do_strip
        self.word_num = 0
        self.word_count = Counter()  # reserved
        self.unknown = unknown
        self.ignore = ignore
        self.pad = pad
        if ignore:
            self.word2idx[ignore] = -100
            self.idx2word[-100] = ignore
            self.word_count[ignore] += int(1e10)
        if pad:
            assert self.word_num == 0, f"The pad id must be 0"
            self.word_count[pad] += int(1e10)
            self.word2idx[pad] = self.word_num
            self.idx2word[self.word_num] = pad
            self.word_num += 1
        if unknown:
            self.word_count[unknown] += int(1e10)
            self.word2idx[unknown] = self.word_num
            self.idx2word[self.word_num] = unknown
            self.word_num += 1

    def dumps(self) -> Dict:
        """dumps the object to dict

        Returns:
            self.__dict__

        """
        return self.__dict__

    def dump(self, path: str):
        """dump the object to path

        Returns:
            self.__dict__

        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, attr: Dict):
        """load the object from dict

        Args:
            attr: self.__dict__

        Returns:
            initialized Vocabulary

        """
        vocab = cls()
        vocab.__dict__ = attr
        return vocab

    @classmethod
    def load_from_file(cls, path: str):
        """load the object from path

        Args:
            path: a json file

        Returns:
            initialized Vocabulary

        """
        with open(path, "r", encoding="utf-8") as f:
            attr = json.load(f)
        vocab = cls()
        vocab.__dict__ = attr
        id2word = vocab.idx2word
        vocab.idx2word = {int(i): id2word[i] for i in id2word}
        return vocab

    def __getitem__(self, index: int):
        """get the token by index

        Args:
            index: token index

        Returns:
            `word` which index is geven, if index is not out of range

        Raises:
            KeyError

        """
        return self.idx2word[int(index)]

    def auto_get_index(self, data: Union[str, List]):
        """get the index of word ∈data from this vocab

        Args:
            data: auto detection

        Returns:
            type the same as data

        """
        if isinstance(data, str):
            return self.get_index(data)
        elif isinstance(data, list):
            return [self.auto_get_index(subdata) for subdata in data]
        else:
            raise ValueError("Don't support the type of {}".format(data))

    def get_index(self, word: str) -> int:
        """get the index of word from this vocab

        Args:
            word: a single token

        Returns:
            index

        """
        if self.do_strip:
            word = word.strip()
        try:
            return self.word2idx[word]
        except:
            if self.unknown:
                return self.word2idx[self.unknown]
            else:
                raise KeyError("Unkown word: {}".format(word))

    def filter_rare(self, min_freq=1, most_common=-1):
        """filter the words which count is to small.

        min_freq and most_common can not set all

        Args:
            min_freq: minist frequency
            most_common: most common number, -1 means all

        Returns:
            None

        """
        self.word2idx = {}
        self.idx2word = {}
        assert (
            min_freq == 1 or most_common == -1
        ), "You should set the min_freq=1 or most_common=-1."
        if most_common != -1:
            for i, (token, freq) in enumerate(self.word_count.most_common(most_common)):
                self.word2idx[token] = i
                self.idx2word[i] = token
        else:
            index = 0
            for token in self.word_count:
                if self.word_count[token] >= min_freq:
                    self.word2idx[token] = index
                    self.idx2word[index] = token
                    index += 1
        return self

    def get_word(self, index: int) -> str:
        """get the word by index

        Args:
            index: word index

        Returns:
            word

        """

        try:
            return self.idx2word[int(index)]
        except:
            if index == -1:
                return "[unknown]"
            raise KeyError("Undefined index: {}".format(index))

    def add(self, word):
        """add one word to vocab

        Args:
            word: single word

        Returns:
            self

        """

        if not self.word_count[word]:
            self.word2idx[word] = self.word_num
            self.idx2word[self.word_num] = word
            self.word_num += 1
        self.word_count[word] += 1
        return self

    def auto_update(self, data: Union[str, Iterable]):
        """auto detect data type to update the vocab

        Args:
            data:  str| List[str] | Set[str] | List[List[str]]

        Returns:
            self

        """
        if isinstance(data, str):
            self.add(data)
        elif (
            isinstance(data, list)
            or isinstance(data, set)
            or isinstance(data, pd.Series)
        ):
            self.add_from_iter(data)
        else:
            raise ValueError("Don't support the type of {}".format(data))
        return self

    def __len__(self):
        """get the token num of vocab
        Returns:
            len(self.word2idx)

        """
        return len(self.word2idx)

    def add_from_iter(self, iterator):
        """add the tokens in iterator to vocab

        Args:
            iterator: List[str] | Set[str] | List[List[str]]

        Returns:
            self

        """
        for word in iterator:
            if isinstance(word, list) or isinstance(word, set):
                self.add_from_iter(word)
            elif isinstance(word, str):
                self.add(word)
            else:
                raise ValueError("Don't support the type of {}".format(word))
        return self
