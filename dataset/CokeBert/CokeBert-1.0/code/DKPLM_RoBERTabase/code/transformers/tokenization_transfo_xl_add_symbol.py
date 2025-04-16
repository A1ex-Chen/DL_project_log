def add_symbol(self, sym):
    if sym not in self.sym2idx:
        self.idx2sym.append(sym)
        self.sym2idx[sym] = len(self.idx2sym) - 1
