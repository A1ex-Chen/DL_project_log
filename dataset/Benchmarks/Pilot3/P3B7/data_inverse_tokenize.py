def inverse_tokenize(self):
    self.vocab.inverse = {v: k for k, v in self.vocab.word2idx.items()}
