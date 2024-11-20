def copy(self):
    new = WeakIdKeyDictionary()
    with _IterationGuard(self):
        for key, value in self.data.items():
            o = key()
            if o is not None:
                new[o] = value
    return new
