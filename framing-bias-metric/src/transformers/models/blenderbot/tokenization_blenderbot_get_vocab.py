def get_vocab(self) ->Dict:
    return dict(self.encoder, **self.added_tokens_encoder)
