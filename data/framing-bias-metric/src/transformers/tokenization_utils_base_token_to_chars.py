def token_to_chars(self, batch_or_token_index: int, token_index: Optional[
    int]=None) ->CharSpan:
    """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a :class:`~transformers.tokenization_utils_base.CharSpan` with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - ``self.token_to_chars(token_index)`` if batch size is 1
        - ``self.token_to_chars(batch_index, token_index)`` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the token or tokens in
                the sequence.

        Returns:
            :class:`~transformers.tokenization_utils_base.CharSpan`: Span of characters in the original string.
        """
    if not self._encodings:
        raise ValueError(
            'token_to_chars() is not available when using Python based tokenizers'
            )
    if token_index is not None:
        batch_index = batch_or_token_index
    else:
        batch_index = 0
        token_index = batch_or_token_index
    return CharSpan(*self._encodings[batch_index].token_to_chars(token_index))
