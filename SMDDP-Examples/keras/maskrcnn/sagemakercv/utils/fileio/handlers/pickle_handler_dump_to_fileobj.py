def dump_to_fileobj(self, obj, file, **kwargs):
    kwargs.setdefault('protocol', 2)
    pickle.dump(obj, file, **kwargs)
