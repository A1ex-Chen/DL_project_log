@property
def n_sequences(self) ->Optional[int]:
    """
        :obj:`Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        :class:`~transformers.BatchEncoding`. Currently can be one of :obj:`None` (unknown), :obj:`1` (a single
        sentence) or :obj:`2` (a pair of sentences)
        """
    return self.n_sequences
