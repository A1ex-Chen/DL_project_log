def get_vocab(self) ->Dict:
    vocab = self.encoder.copy()
    vocab.update(self.added_tokens_encoder)
    return vocab
