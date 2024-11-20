def append(self, pred_sent, ref_sents):
    """Append a sentence for measurements, increase counters.

        @param pred_sent: the system output sentence (string/list of tokens)
        @param ref_sents: the corresponding reference sentences (list of strings/lists of tokens)
        """
    pred_sent, ref_sents = self.check_tokenized(pred_sent, ref_sents)
    for n in range(self.max_ngram):
        self.cand_lens[n].append(len(pred_sent) - n)
        merged_ref_ngrams = self.get_ngram_counts(n + 1, ref_sents)
        pred_ngrams = self.get_ngram_counts(n + 1, [pred_sent])
        hit_ngrams = {}
        for ngram in pred_ngrams:
            hits = min(pred_ngrams[ngram], merged_ref_ngrams.get(ngram, 0))
            if hits:
                hit_ngrams[ngram] = hits
        self.hit_ngrams[n].append(hit_ngrams)
        for ref_sent in ref_sents:
            for ngram in self.ngrams(n + 1, ref_sent):
                self.ref_ngrams[n + 1][ngram] += 1
    ref_len_sum = sum(len(ref_sent) for ref_sent in ref_sents)
    self.ref_ngrams[0][()] += ref_len_sum
    self.avg_ref_len += ref_len_sum / float(len(ref_sents))
