def token_to_sequence(self, batch_or_token_index: int, token_index:
    Optional[int]=None) ->int:
    """
        Get the index of the sequence represented by the given token. In the general use case, this method returns
        :obj:`0` for a single sequence or the first sequence of a pair, and :obj:`1` for the second sequence of a pair

        Can be called as:

        - ``self.token_to_sequence(token_index)`` if batch size is 1
        - ``self.token_to_sequence(batch_index, token_index)`` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the token in the sequence.
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the token in the
                sequence.

        Returns:
            :obj:`int`: Index of the word in the input sequence.
        """
    if not self._encodings:
        raise ValueError(
            'token_to_sequence() is not available when using Python based tokenizers'
            )
    if token_index is not None:
        batch_index = batch_or_token_index
    else:
        batch_index = 0
        token_index = batch_or_token_index
    if batch_index < 0:
        batch_index = self._batch_size + batch_index
    if token_index < 0:
        token_index = self._seq_len + token_index
    return self._encodings[batch_index].token_to_sequence(token_index)
