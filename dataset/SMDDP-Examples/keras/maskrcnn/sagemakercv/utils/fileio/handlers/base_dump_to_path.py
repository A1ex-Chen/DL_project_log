def dump_to_path(self, obj, filepath, mode='w', **kwargs):
    with open(filepath, mode) as f:
        self.dump_to_fileobj(obj, f, **kwargs)
