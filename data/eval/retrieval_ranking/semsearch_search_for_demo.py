def search_for_demo(self, query, candidates, token_sentences, top_n=10,
    contextual=False):
    """ """
    self.logger.debug('number of candidates: %d', len(candidates))
    if not contextual:
        phrases_only = [phrase for phrase, _, _ in candidates]
        scores = self.scorer.score_batch(query, phrases_only)
        results = [{'phrase': phrase, 'score': score} for phrase, score in
            zip(candidates, scores)]
    else:
        scores = self.scorer.score_batch_contextual(query, candidates,
            token_sentences, self.list_oracle, max_seq_length=512,
            window_size=-1, use_context_query=True)
        results = [{'phrase': phrase, 'score': score} for phrase, score in
            zip(candidates, scores)]
    results = sorted(results, reverse=True, key=lambda x: x['score'])[:top_n]
    return results
