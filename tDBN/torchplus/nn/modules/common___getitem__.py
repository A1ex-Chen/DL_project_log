def __getitem__(self, idx):
    if not -len(self) <= idx < len(self):
        raise IndexError('index {} is out of range'.format(idx))
    if idx < 0:
        idx += len(self)
    it = iter(self._modules.values())
    for i in range(idx):
        next(it)
    return next(it)
