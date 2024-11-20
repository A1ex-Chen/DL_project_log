def get_ngram_counts(self, n, sents):
    """Returns a dictionary with counts of all n-grams in the given sentences.
        @param n: the "n" in n-grams (how long the n-grams should be)
        @param sents: list of sentences for n-gram counting
        @return: a dictionary (ngram: count) listing counts of n-grams attested in any of the sentences
        """
    merged_ngrams = {}
    for sent in sents:
        ngrams = defaultdict(int)
        for ngram in self.ngrams(n, sent):
            ngrams[ngram] += 1
        for ngram, cnt in ngrams.items():
            merged_ngrams[ngram] = max((merged_ngrams.get(ngram, 0), cnt))
    return merged_ngrams
