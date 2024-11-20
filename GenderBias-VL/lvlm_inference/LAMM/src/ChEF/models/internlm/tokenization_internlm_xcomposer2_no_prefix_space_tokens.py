@property
def no_prefix_space_tokens(self):
    if self._no_prefix_space_tokens is None:
        vocab = self.convert_ids_to_tokens(list(range(self.vocab_size)))
        self._no_prefix_space_tokens = {i for i, tok in enumerate(vocab) if
            not tok.startswith('▁')}
    return self._no_prefix_space_tokens
