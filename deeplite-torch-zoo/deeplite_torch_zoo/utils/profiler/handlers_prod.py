def prod(iterable):
    return reduce(operator.mul, iterable, 1)
