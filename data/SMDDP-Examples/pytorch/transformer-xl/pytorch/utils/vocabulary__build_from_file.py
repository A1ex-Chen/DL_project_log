def _build_from_file(self, vocab_file):
    self.idx2sym = []
    self.sym2idx = OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            symb = line.strip().split()[0]
            self.add_symbol(symb)
    self.unk_idx = self.sym2idx['<UNK>']
