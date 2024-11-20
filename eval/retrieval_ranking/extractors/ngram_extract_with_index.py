def extract_with_index(self, sent, ngram_min=None, ngram_max=None):
    phrases = []
    if ngram_min and ngram_max:
        for n in range(ngram_min, ngram_max + 1):
            phrases.extend(self._get_ngrams_with_index(sent, n))
    else:
        for n in range(self.ngram_min, self.ngram_max + 1):
            phrases.extend(self._get_ngrams_with_index(sent, n))
    phrases = [x for x in phrases if len(x) != 0]
    return list(phrases)
