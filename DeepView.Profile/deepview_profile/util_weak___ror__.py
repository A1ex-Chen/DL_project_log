def __ror__(self, other):
    if isinstance(other, _collections_abc.Mapping):
        c = self.__class__()
        c.update(other)
        c.update(self)
        return c
    return NotImplemented
