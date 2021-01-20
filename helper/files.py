from typing import Iterator, List


def read_lines(path: str) -> Iterator[str]:
    with open(path) as f:
        for line in f:
            yield line[:-1]


def read_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    return lines
