def __setattr__(self, name, value):
    if isinstance(value, dict):
        value = ConfigDict(value)
    self._cfg_dict.__setattr__(name, value)
