def __getitem__(self, key):
    return self.data[WeakIdRef(key)]