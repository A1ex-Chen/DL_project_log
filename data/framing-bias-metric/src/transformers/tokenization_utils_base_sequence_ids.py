def sequence_ids(self, batch_index: int=0) ->List[Optional[int]]:
    """
        Return a list mapping the tokens to the id of their original sentences:

            - :obj:`None` for special tokens added around or between sequences,
            - :obj:`0` for tokens corresponding to words in the first sequence,
            - :obj:`1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
              encoded.

        Args:
            batch_index (:obj:`int`, `optional`, defaults to 0): The index to access in the batch.

        Returns:
            :obj:`List[Optional[int]]`: A list indicating the sequence id corresponding to each token. Special tokens
            added by the tokenizer are mapped to :obj:`None` and other tokens are mapped to the index of their
            corresponding sequence.
        """
    if not self._encodings:
        raise ValueError(
            'sequence_ids() is not available when using Python-based tokenizers'
            )
    return self._encodings[batch_index].sequence_ids
