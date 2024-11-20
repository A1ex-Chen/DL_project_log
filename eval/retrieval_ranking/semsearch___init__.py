def __init__(self, use_cache=False):
    self.logger = CreateLogger()
    self.scorer = None
    self.logger.info('REMOVE_CANDIDATE_STOPWORDS: %s',
        REMOVE_CANDIDATE_STOPWORDS)
    self.sentences = []
    self.phrases = list()
    self.contextual = False
    self.list_oracle = []
    self.use_cache = use_cache
    self.dic_cache_candidates = {}
    self.logger.info('REMOVE_CANDIDATE_STOPWORDS: %s',
        REMOVE_CANDIDATE_STOPWORDS)
    self.logger.debug('[use_cache]: %s', use_cache)
    self.nlp = spacy.load('en_core_web_sm')
