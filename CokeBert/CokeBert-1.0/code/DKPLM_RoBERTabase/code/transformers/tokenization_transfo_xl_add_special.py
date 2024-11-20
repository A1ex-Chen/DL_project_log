def add_special(self, sym):
    if sym not in self.sym2idx:
        self.idx2sym.append(sym)
        self.sym2idx[sym] = len(self.idx2sym) - 1
        setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])
