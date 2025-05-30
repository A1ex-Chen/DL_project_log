def get_vocab(self):
    """Returns vocab as a dict."""
    vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
    vocab.update(self.added_tokens_encoder)
    return vocab
