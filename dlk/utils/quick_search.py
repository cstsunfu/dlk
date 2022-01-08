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

import ahocorasick
from typing import Iterable, List, Dict

class QuickSearch(object):
    """Ahocorasick enhanced Trie"""
    def __init__(self, words: Iterable=[]):
        """init the QuickSearch

        Args:
            words: tokens in use words to update the trie

        Returns: 
            None

        """
        super(QuickSearch, self).__init__()

        if not isinstance(words, Iterable) :
            raise PermissionError("The words must be a Iterable object.")
        self.next_index = 0

        self.ac = ahocorasick.Automaton()

        for sub_str in words:
            sub_str = f" {sub_str.strip()} "
            self.ac.add_word(sub_str, (self.next_index, sub_str))
            self.next_index += 1
        self.ac.make_automaton()

    def search(self, search_str:str)->List[Dict]:
        """find whether some sub_str in trie

        Args:
            search_str: find the search_str

        Returns: 
            >>> list of result:
            >>> the result organized as {
            >>>     "start": start_position,
            >>>     "end": end_position,
            >>>     "str": search_str[start_position: end_position]
            >>> }

        """
        result = []
        for end_index, (_, original_value) in self.ac.iter(search_str):
            start_index = end_index - len(original_value) + 1
            end_index = end_index + 1
            result.append({"start": start_index, "end": end_index, "str": original_value})
        return result

    def has(self, key: str)->bool:
        """check key is in trie

        Args:
            key: a token(str)

        Returns: 
            bool(has or not)

        """
        try:
            self.ac.get(key)
        except:
            return False
        return True

    def add_word(self, word: str):
        """add a single word to trie

        Args:
            word: single token

        Returns: 
            None

        """
        self.ac.add_word(word, (self.next_index, word))
        self.next_index += 1
        self.ac.make_automaton()

    def add_words(self, words: Iterable):
        """add words from iterator to the trie

        Args:
            words: Iterable[tokens]

        Returns: 
            None

        """
        for sub_str in words:
            self.ac.add_word(sub_str, (self.next_index, sub_str))
            self.next_index += 1
        self.ac.make_automaton()
