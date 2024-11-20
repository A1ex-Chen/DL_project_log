def distinct_n(sentences, n):
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences
        ) / len(sentences)
