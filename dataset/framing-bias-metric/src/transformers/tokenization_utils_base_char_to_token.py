def char_to_token(self, batch_or_char_index: int, char_index: Optional[int]
    =None, sequence_index: int=0) ->int:
    """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - ``self.char_to_token(char_index)`` if batch size is 1
        - ``self.char_to_token(batch_index, char_index)`` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index of the word in the
                sequence.
            sequence_index (:obj:`int`, `optional`, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            :obj:`int`: Index of the token.
        """
    if not self._encodings:
        raise ValueError(
            'char_to_token() is not available when using Python based tokenizers'
            )
    if char_index is not None:
        batch_index = batch_or_char_index
    else:
        batch_index = 0
        char_index = batch_or_char_index
    return self._encodings[batch_index].char_to_token(char_index,
        sequence_index)
