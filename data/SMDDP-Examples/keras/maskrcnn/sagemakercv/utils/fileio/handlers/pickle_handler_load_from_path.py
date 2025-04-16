def load_from_path(self, filepath, **kwargs):
    return super(PickleHandler, self).load_from_path(filepath, mode='rb',
        **kwargs)
