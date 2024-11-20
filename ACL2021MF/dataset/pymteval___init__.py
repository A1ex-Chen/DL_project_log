def __init__(self, max_ngram=5, case_sensitive=False):
    """Create the scoring object.
        @param max_ngram: the n-gram level to compute the score for (default: 5)
        @param case_sensitive: use case-sensitive matching (default: no)
        """
    super(NISTScore, self).__init__(max_ngram, case_sensitive)
    self.reset()
