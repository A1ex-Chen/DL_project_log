def reset(self):
    """Reset the object, zero all counters."""
    self.ref_ngrams = [defaultdict(int) for _ in range(self.max_ngram + 1)]
    self.hit_ngrams = [[] for _ in range(self.max_ngram)]
    self.cand_lens = [[] for _ in range(self.max_ngram)]
    self.avg_ref_len = 0.0
