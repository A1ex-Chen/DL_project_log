def parse(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, n))
