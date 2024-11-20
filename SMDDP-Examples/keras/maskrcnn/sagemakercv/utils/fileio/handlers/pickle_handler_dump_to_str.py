def dump_to_str(self, obj, **kwargs):
    kwargs.setdefault('protocol', 2)
    return pickle.dumps(obj, **kwargs)
