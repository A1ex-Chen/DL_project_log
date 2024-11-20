def tokens(self, batch_index: int=0) ->List[str]:
    """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (:obj:`int`, `optional`, defaults to 0): The index to access in the batch.

        Returns:
            :obj:`List[str]`: The list of tokens at that index.
        """
    if not self._encodings:
        raise ValueError(
            'tokens() is not available when using Python-based tokenizers')
    return self._encodings[batch_index].tokens
