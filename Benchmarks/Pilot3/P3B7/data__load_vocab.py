def _load_vocab(self):
    self.vocab = torch.load(self.root.joinpath('vocab.pt'))
    self.num_vocab = len(self.vocab)
