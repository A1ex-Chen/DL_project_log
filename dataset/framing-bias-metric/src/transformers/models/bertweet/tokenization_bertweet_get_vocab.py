def get_vocab(self):
    return dict(self.encoder, **self.added_tokens_encoder)
