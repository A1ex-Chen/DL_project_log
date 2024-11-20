def score_batch(self, query, list_phrase, list_oracle=None):
    """ """
    list_phrase = list_phrase + (list_oracle if list_oracle else [])
    query_emb = self.embed_batch([query])
    phrase_emb = self.embed_batch(list_phrase)
    score = cosine_similarity(query_emb, phrase_emb)
    return score.tolist()[-1]
