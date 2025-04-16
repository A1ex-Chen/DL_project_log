def get_vocab(self):
    return dict(self.vocab, **self.added_tokens_encoder)
