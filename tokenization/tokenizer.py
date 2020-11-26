from typing import List
import abc

from tokenization.token import Token


class Tokenizer:
    @abc.abstractmethod
    def tokenize(self, sequence: str) -> List[Token]:
        raise NotImplementedError()
