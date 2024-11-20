def get_src_vocab(self):
    return dict(self.encoder, **self.added_tokens_encoder)
