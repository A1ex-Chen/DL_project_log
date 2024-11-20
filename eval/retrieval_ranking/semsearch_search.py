def search(self, query, phrases=None, top_n=10, window_size=0):
    """ """
    self.logger.debug('number of candidates: %d', len(self.phrases))
    if hasattr(self.scorer.__class__, 'score_batch') and callable(getattr(
        self.scorer.__class__, 'score_batch')):
        if phrases is None:
            phrases = list(self.phrases)
            if len(phrases) == 0:
                phrases = ['NA']
        if not self.contextual:
            scores = self.scorer.score_batch(query, phrases, self.list_oracle)
            phrases.extend(self.list_oracle)
            results = [{'phrase': phrase, 'score': score} for phrase, score in
                zip(phrases, scores)]
        else:
            scores = self.scorer.score_batch_contextual(query, phrases,
                self.sentences, self.list_oracle, self.max_seq_length,
                window_size=window_size, use_context_query=True)
            phrases.extend(self.list_oracle)
            results = [{'phrase': phrase[0], 'score': score} for phrase,
                score in zip(phrases, scores)]
    else:
        results = [{'phrase': phrase, 'score': self.scorer.score(query,
            phrase)} for phrase in self.phrases]
    results = sorted(results, reverse=True, key=lambda x: x['score'])[:top_n]
    return results
