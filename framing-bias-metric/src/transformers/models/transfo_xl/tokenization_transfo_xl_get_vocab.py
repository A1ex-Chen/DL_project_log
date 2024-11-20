def get_vocab(self):
    return dict(self.sym2idx, **self.added_tokens_encoder)
