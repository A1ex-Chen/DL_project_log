def update(self, dict=None, **kwargs):
    d = self.data
    if dict is not None:
        if not hasattr(dict, 'items'):
            dict = type({})(dict)
        for key, value in dict.items():
            d[WeakIdRef(key, self._remove)] = value
    if len(kwargs):
        self.update(kwargs)
