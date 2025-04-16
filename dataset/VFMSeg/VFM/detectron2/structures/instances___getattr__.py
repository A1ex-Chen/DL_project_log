def __getattr__(self, name: str) ->Any:
    if name == '_fields' or name not in self._fields:
        raise AttributeError("Cannot find field '{}' in the given Instances!"
            .format(name))
    return self._fields[name]
