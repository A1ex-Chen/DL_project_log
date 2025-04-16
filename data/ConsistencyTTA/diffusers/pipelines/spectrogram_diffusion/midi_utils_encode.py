def encode(self, token_ids):
    encoded = []
    for token_id in token_ids:
        if not 0 <= token_id < self._num_regular_tokens:
            raise ValueError(
                f'token_id {token_id} does not fall within valid range of [0, {self._num_regular_tokens})'
                )
        encoded.append(token_id + self._num_special_tokens)
    encoded.append(1)
    encoded = encoded + [0] * (INPUT_FEATURE_LENGTH - len(encoded))
    return encoded
