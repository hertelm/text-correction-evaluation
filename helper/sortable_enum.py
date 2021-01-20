from enum import Enum


class SortableEnum(Enum):
    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return self.name
