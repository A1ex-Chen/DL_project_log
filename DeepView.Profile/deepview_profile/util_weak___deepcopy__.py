def __deepcopy__(self, memo):
    from copy import deepcopy
    new = self.__class__()
    with _IterationGuard(self):
        for key, value in self.data.items():
            o = key()
            if o is not None:
                new[o] = deepcopy(value, memo)
    return new
