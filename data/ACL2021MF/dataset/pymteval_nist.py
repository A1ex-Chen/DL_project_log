def nist(self):
    """Return the current NIST score, according to the accumulated counts."""
    hit_infos = [(0.0) for _ in range(self.max_ngram)]
    for n in range(self.max_ngram):
        for hit_ngrams in self.hit_ngrams[n]:
            hit_infos[n] += sum(self.info(ngram) * hits for ngram, hits in
                hit_ngrams.items())
    total_lens = [sum(self.cand_lens[n]) for n in range(self.max_ngram)]
    nist_sum = sum(old_div(hit_info, total_len) for hit_info, total_len in
        zip(hit_infos, total_lens))
    bp = self.nist_length_penalty(sum(self.cand_lens[0]), self.avg_ref_len)
    return bp * nist_sum
