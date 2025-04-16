def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)
