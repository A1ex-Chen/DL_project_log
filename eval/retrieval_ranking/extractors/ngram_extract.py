def extract(self, text):
    """ """
    phrases = []
    for n in range(self.ngram_min, self.ngram_max + 1):
        phrases.extend(self._get_ngrams(text, n))
    phrases = [x for x in phrases if len(x) != 0]
    return list(phrases)
