def set_extractor(self, extractor, ngram_min, ngram_max):
    """ """
    if extractor == 'ngrams':
        instantiated_extractor = NGramExtractor(ngram_min, ngram_max,
            REMOVE_CANDIDATE_STOPWORDS)
    else:
        assert False
    self.extractor = instantiated_extractor
