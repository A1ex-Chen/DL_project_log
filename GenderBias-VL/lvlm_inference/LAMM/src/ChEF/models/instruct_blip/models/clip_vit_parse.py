def parse(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, n))
