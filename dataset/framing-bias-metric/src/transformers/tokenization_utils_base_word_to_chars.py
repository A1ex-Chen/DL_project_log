def word_to_chars(self, batch_or_word_index: int, word_index: Optional[int]
    =None, sequence_index: int=0) ->CharSpan:
    """
        Get the character span in the original string corresponding to given word in a sequence of the batch.

        Character spans are returned as a CharSpan NamedTuple with:

        - start: index of the first character in the original string
        - end: index of the character following the last character in the original string

        Can be called as:

        - ``self.word_to_chars(word_index)`` if batch size is 1
        - ``self.word_to_chars(batch_index, word_index)`` if batch size is greater or equal to 1

        Args:
            batch_or_word_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            word_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the word in the
                sequence.
            sequence_index (:obj:`int`, `optional`, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            :obj:`CharSpan` or :obj:`List[CharSpan]`: Span(s) of the associated character or characters in the string.
            CharSpan are NamedTuple with:

                - start: index of the first character associated to the token in the original string
                - end: index of the character following the last character associated to the token in the original
                  string
        """
    if not self._encodings:
        raise ValueError(
            'word_to_chars() is not available when using Python based tokenizers'
            )
    if word_index is not None:
        batch_index = batch_or_word_index
    else:
        batch_index = 0
        word_index = batch_or_word_index
    return CharSpan(*self._encodings[batch_index].word_to_chars(word_index,
        sequence_index))
