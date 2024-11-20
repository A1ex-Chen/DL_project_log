def _build_from_file(self, vocab_file):
    self.idx2sym = []
    self.sym2idx = OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            symb = line.strip().split()[0]
            self.add_symbol(symb)
    if '<UNK>' in self.sym2idx:
        self.unk_idx = self.sym2idx['<UNK>']
    elif '<unk>' in self.sym2idx:
        self.unk_idx = self.sym2idx['<unk>']
    else:
        raise ValueError('No <unkown> token in vocabulary')
