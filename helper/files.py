from typing import Iterator


def read_lines(path: str) -> Iterator[str]:
    with open(path) as f:
        for line in f:
            yield line[:-1]
