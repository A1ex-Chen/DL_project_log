def bleu(self):
    """Return the current BLEU score, according to the accumulated counts."""
    bp = 1.0
    if self.cand_lens[0] <= self.ref_len:
        bp = math.exp(1.0 - old_div(self.ref_len, float(self.cand_lens[0]) if
            self.cand_lens[0] else 1e-05))
    return bp * self.ngram_precision()
