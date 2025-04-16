def info(self, ngram):
    """Return the NIST informativeness of an n-gram."""
    if ngram not in self.ref_ngrams[len(ngram)]:
        return 0.0
    return math.log(self.ref_ngrams[len(ngram) - 1][ngram[:-1]] / float(
        self.ref_ngrams[len(ngram)][ngram]), 2)
