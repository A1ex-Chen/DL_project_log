def word_ids(self, batch_index: int=0) ->List[Optional[int]]:
    """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (:obj:`int`, `optional`, defaults to 0): The index to access in the batch.

        Returns:
            :obj:`List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by
            the tokenizer are mapped to :obj:`None` and other tokens are mapped to the index of their corresponding
            word (several tokens will be mapped to the same word index if they are parts of that word).
        """
    if not self._encodings:
        raise ValueError(
            'word_ids() is not available when using Python-based tokenizers')
    return self._encodings[batch_index].word_ids
