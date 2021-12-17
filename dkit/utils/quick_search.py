import ahocorasick
from typing import Iterable, List, Dict

class QuickSearch(object):
    """docstring for QuickSearch"""
    def __init__(self, words: Iterable=[]):
        """
            words: use words to init the QuickSearch
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
        """find whether some sub_str in QuickSearch
        :search_str: TODO
        :returns: List of result
            the result organized as {
                "start": start_position,
                "end": end_position,
                "str": search_str[start_position: end_position]
            }
        """
        result = []
        for end_index, (_, original_value) in self.ac.iter(search_str):
            start_index = end_index - len(original_value) + 1
            end_index = end_index + 1
            result.append({"start": start_index, "end": end_index, "str": original_value})
        return result

    def has(self, key: str)->bool:
        """find key is in our data
        :key: str
        :returns: true|false

        """
        try:
            self.ac.get(key)
        except:
            return False
        return True

    def add_word(self, word: str):
        """add a single word to QuickSearch
        :word: str
        :returns: None
        """
        self.ac.add_word(word, (self.next_index, word))
        self.next_index += 1
        self.ac.make_automaton()

    def add_words(self, words: Iterable=[]):
        """add words from iterator to the ac
        :words: Iterable
        :returns: None
        """
        for sub_str in words:
            self.ac.add_word(sub_str, (self.next_index, sub_str))
            self.next_index += 1
        self.ac.make_automaton()
