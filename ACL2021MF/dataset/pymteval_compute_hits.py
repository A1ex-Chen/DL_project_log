def compute_hits(self, n, pred_sent, ref_sents):
    """Compute clipped n-gram hits for the given sentences and the given N

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param pred_sent: the system output sentence (tree/tokens)
        @param ref_sents: the corresponding reference sentences (list/tuple of trees/tokens)
        """
    merged_ref_ngrams = self.get_ngram_counts(n, ref_sents)
    pred_ngrams = self.get_ngram_counts(n, [pred_sent])
    hits = 0
    for ngram, cnt in pred_ngrams.items():
        hits += min(merged_ref_ngrams.get(ngram, 0), cnt)
    return hits
