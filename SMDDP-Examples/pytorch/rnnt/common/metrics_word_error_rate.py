def word_error_rate(hypotheses, references):
    """Computes average Word Error Rate (WER) between two text lists."""
    scores = 0
    words = 0
    len_diff = len(references) - len(hypotheses)
    if len_diff > 0:
        raise ValueError(
            'Uneqal number of hypthoses and references: {0} and {1}'.format
            (len(hypotheses), len(references)))
    elif len_diff < 0:
        hypotheses = hypotheses[:len_diff]
    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer, scores, words
