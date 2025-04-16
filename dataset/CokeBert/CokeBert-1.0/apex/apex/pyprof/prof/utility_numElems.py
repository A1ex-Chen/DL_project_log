@staticmethod
def numElems(shape):
    assert type(shape) == tuple
    return reduce(lambda x, y: x * y, shape, 1)
