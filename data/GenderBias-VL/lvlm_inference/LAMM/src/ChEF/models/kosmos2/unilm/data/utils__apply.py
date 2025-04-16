def _apply(x):
    if isinstance(x, np.ndarray):
        return f(x)
    elif isinstance(x, collections.OrderedDict):
        od = collections.OrderedDict((key, _apply(value)) for key, value in
            x.items())
        od.__dict__ = x.__dict__
        return od
    elif isinstance(x, dict):
        return {key: _apply(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [_apply(x) for x in x]
    elif isinstance(x, tuple):
        return tuple(_apply(x) for x in x)
    elif isinstance(x, set):
        return {_apply(x) for x in x}
    else:
        return x
