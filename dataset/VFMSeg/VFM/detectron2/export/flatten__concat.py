@staticmethod
def _concat(values):
    ret = ()
    sizes = []
    for v in values:
        assert isinstance(v, tuple), 'Flattened results must be a tuple'
        ret = ret + v
        sizes.append(len(v))
    return ret, sizes
