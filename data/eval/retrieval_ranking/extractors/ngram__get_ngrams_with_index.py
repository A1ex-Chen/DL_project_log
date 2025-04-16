def _get_ngrams_with_index(self, text, n):
    list_ngram = []
    tokens = [token.text.lower() for token in self.nlp.tokenizer(text)]
    words_indices = [(word, i) for i, word in enumerate(tokens)]
    ngrams_indices = [grams for grams in ngrams(words_indices, n)]
    for grams_indices in ngrams_indices:
        phrase, indices = [], []
        for gram, idx in grams_indices:
            phrase.append(gram)
            indices.append(idx)
        list_ngram.append((' '.join(phrase), indices))
    return list_ngram
