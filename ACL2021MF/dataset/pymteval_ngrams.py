def ngrams(self, n, sent):
    """Given a sentence, return n-grams of nodes for the given N. Lowercases
        everything if the measure should not be case-sensitive.

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param sent: the sent in question
        @return: n-grams of nodes, as tuples of tuples (t-lemma & formeme)
        """
    if not self.case_sensitive:
        return list(zip(*[[tok.lower() for tok in sent[i:]] for i in range(n)])
            )
    return list(zip(*[sent[i:] for i in range(n)]))
