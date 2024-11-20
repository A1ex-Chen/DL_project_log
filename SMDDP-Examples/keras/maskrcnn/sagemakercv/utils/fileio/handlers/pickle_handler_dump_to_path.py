def dump_to_path(self, obj, filepath, **kwargs):
    super(PickleHandler, self).dump_to_path(obj, filepath, mode='wb', **kwargs)
