def _get_ngrams(self, text, n):
    tokens = [token.text.lower() for token in self.nlp.tokenizer(text)]
    list_ngram = [' '.join(grams).replace(" 's", "'s").replace(' - ', '-').
        replace(' / ', '/').replace(' %', '%') for grams in ngrams(tokens, n)]
    return list_ngram
