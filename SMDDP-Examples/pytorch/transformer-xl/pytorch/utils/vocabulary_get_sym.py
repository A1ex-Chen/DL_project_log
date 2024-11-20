def get_sym(self, idx):
    assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
    return self.idx2sym[idx]
