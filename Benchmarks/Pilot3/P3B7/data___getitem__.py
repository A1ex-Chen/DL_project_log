def __getitem__(self, idx):
    return self.data[idx], self._index_target(idx)
