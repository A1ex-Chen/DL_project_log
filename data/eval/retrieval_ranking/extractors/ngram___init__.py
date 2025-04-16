def __init__(self, ngram_min=2, ngram_max=3, remove_stopword=False):
    """ """
    self.logger = CreateLogger()
    self.ngram_min = ngram_min
    self.ngram_max = ngram_max
    self.remove_stopword = remove_stopword
    self.nlp = spacy.load('en_core_web_sm')
    self.logger.info('ngrams (%d)-(%d)', self.ngram_min, self.ngram_max)
