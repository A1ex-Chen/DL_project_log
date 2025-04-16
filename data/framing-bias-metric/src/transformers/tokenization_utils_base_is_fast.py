@property
def is_fast(self) ->bool:
    """
        :obj:`bool`: Indicate whether this :class:`~transformers.BatchEncoding` was generated from the result of a
        :class:`~transformers.PreTrainedTokenizerFast` or not.
        """
    return self._encodings is not None
