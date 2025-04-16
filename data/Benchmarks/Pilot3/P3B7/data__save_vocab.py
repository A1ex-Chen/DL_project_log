def _save_vocab(self, vocab):
    torch.save(vocab, self.root.joinpath('vocab.pt'))
