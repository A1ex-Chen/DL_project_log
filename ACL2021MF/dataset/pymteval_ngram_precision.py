def ngram_precision(self):
    """Return the current n-gram precision (harmonic mean of n-gram precisions up to max_ngram)
        according to the accumulated counts."""
    prec_log_sum = 0.0
    for n_hits, n_len in zip(self.hits, self.cand_lens):
        n_hits += self.smoothing
        n_len += self.smoothing
        n_hits = max(n_hits, self.TINY)
        n_len = max(n_len, self.SMALL)
        prec_log_sum += math.log(old_div(n_hits, n_len))
    return math.exp(1.0 / self.max_ngram * prec_log_sum)
