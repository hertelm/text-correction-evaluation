from typing import List

import re

from tokenization.tokenizer import Tokenizer
from tokenization.token import Token


class RegexTokenizer(Tokenizer):
    def __init__(self):
        # \d+[.,][\d.,]*\d numbers separated by point or comma, first and last characters must be digits
        # \w[\w'-]+\w words separated by hyphens or internal apostrophs
        # \w+ words
        # \S special characters, no whitespaces
        self.pattern = re.compile("\d+[.,][\d.,]*\d|\w[\w'-]+\w|\w+|\S")
        self.number_pattern = re.compile("\d")
        self.word_pattern = re.compile("\w[\w'-]+\w|\w+")

    def words(self, sequence) -> List[str]:
        return self.pattern.findall(sequence)

    def tokenize(self, sequence) -> List[Token]:
        matches = self.pattern.finditer(sequence)
        tokens = [Token(match.group(), match.start(), match.end()) for match in matches]
        return tokens

    def editable(self, word: str) -> bool:
        return self.number_pattern.search(word) is None and self.word_pattern.fullmatch(word) is not None
