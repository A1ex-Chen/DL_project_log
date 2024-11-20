def load_from_path(self, filepath, mode='r', **kwargs):
    with open(filepath, mode) as f:
        return self.load_from_fileobj(f, **kwargs)
