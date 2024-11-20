def __getattr__(self, name):
    return getattr(self._cfg_dict, name)
