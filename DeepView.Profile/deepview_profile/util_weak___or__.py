def __or__(self, other):
    if isinstance(other, _collections_abc.Mapping):
        c = self.copy()
        c.update(other)
        return c
    return NotImplemented
