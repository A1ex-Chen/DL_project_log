def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.gpt2_tokenizer.sym(index
        ) if index < self.vocab_size else self.unk_token
