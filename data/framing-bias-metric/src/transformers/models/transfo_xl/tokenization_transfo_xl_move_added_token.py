def move_added_token(self, token: str, target_idx: int):
    """
        Moves an added token to a specific position in the vocab. This method should be used when resizing an embedding
        layer other than the last one in the `AdaptiveEmbedding` in order to move the token in the tokenizer from the
        default position (at the very end) to the desired one.

        Args:
            token: The token to move to a specific position in the vocab.
            target_idx: The position where the token should be moved to.
        """
    assert token in self.added_tokens_encoder, 'Token which should be moved has to be an added token'
    assert token not in self.idx2sym, 'Token which should be moved is already in vocab'
    self.idx2sym.insert(target_idx, token)
    self.sym2idx[token] = target_idx
    for idx in range(target_idx + 1, len(self.idx2sym)):
        current_sym = self.idx2sym[idx]
        self.sym2idx[current_sym] = idx
    old_index = self.added_tokens_encoder[token]
    del self.added_tokens_decoder[old_index]
    del self.added_tokens_encoder[token]
