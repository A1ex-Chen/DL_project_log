def __getattr__(self, attr):
    if attr not in ['_obj']:
        return getattr(self._obj, attr)
    return getattr(self, attr)
