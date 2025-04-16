def build_vocab(self):
    if self.vocab_file:
        logger.info('building vocab from {}'.format(self.vocab_file))
        self._build_from_file(self.vocab_file)
        logger.info('final vocab size {}'.format(len(self)))
    else:
        logger.info('building vocab with min_freq={}, max_size={}'.format(
            self.min_freq, self.max_size))
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        for sym in self.special:
            self.add_special(sym)
        for sym, cnt in self.counter.most_common(self.max_size):
            if cnt < self.min_freq:
                break
            self.add_symbol(sym)
        logger.info('final vocab size {} from {} unique tokens'.format(len(
            self), len(self.counter)))
