from itertools import count


def izip(*args):
    return zip(count(), *args)
