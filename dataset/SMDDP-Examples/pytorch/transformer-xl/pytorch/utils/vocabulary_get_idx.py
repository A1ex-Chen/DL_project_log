def get_idx(self, sym):
    if sym in self.sym2idx:
        return self.sym2idx[sym]
    else:
        assert '<eos>' not in sym
        assert hasattr(self, 'unk_idx')
        return self.sym2idx.get(sym, self.unk_idx)
