def get_vocab(self):
    vocab = self.vocab.copy()
    vocab.update(self.get_added_vocab())
    return vocab
