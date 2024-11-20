def set_scorer(self, scorer, model_fpath='', scorer_type=''):
    """ """
    if scorer == 'BERT':
        instantiated_scorer = BertScorer(scorer_type, model_fpath)
    elif scorer in ['SentenceBERT', 'PhraseBERT']:
        instantiated_scorer = SentenceBertScorer(scorer_type, model_fpath)
    elif scorer == 'SpanBERT':
        instantiated_scorer = SpanBertScorer(scorer_type, model_fpath)
    elif scorer == 'SimCSE':
        instantiated_scorer = SimCSEScorer(scorer_type, model_fpath)
    elif scorer == 'DensePhrases':
        instantiated_scorer = DensePhrasesScorer(scorer_type, model_fpath)
    elif scorer == 'USE':
        instantiated_scorer = USEScorer(scorer_type)
    else:
        assert False
    self.scorer = instantiated_scorer
