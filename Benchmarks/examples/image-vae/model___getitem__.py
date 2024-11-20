def __getitem__(self, idx):
    if idx < 0 or idx >= len(self._modules):
        raise IndexError('index {} is out of range'.format(idx))
    it = iter(self._modules.values())
    for i in range(idx):
        next(it)
    return next(it)
