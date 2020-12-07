from typing import List


class Token:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end

    def range(self):
        return self.start, self.end

    def __str__(self):
        return str((self.word, self.start, self.end))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.word == other.word and self.start == other.start and self.end == other.end


def recombine_words(words: List[str],
                    original_tokens: List[Token],
                    original_sequence: str):
    recombined_sequence = ""
    for i, (word, token) in enumerate(zip(words, original_tokens)):
        if i > 0:
            recombined_sequence += original_sequence[original_tokens[i - 1].end:token.start]
        recombined_sequence += word
    return recombined_sequence
