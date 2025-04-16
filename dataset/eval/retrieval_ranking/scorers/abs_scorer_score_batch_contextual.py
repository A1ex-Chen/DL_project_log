def score_batch_contextual(self, query, list_phrase, sentences, list_oracle
    =None, max_seq_length=128, window_size=-1, use_context_query=True):
    """ """
    context_phrases, context_queries = [], []
    list_phrase = list_phrase + (list_oracle if list_oracle else [])
    for phrase, index, sent_idx in list_phrase:
        if isinstance(sentences[sent_idx], str):
            tokens = [token.text.lower() for token in self.nlp.tokenizer(
                sentences[sent_idx])]
        else:
            tokens = sentences[sent_idx]
        if window_size >= 0:
            left_context = index[0] - window_size if index[0
                ] - window_size > 0 else 0
            right_context = index[-1] + window_size + 1 if index[-1
                ] + window_size + 1 < len(tokens) else len(tokens)
        else:
            left_context = 0
            right_context = len(tokens)
        context_phrase = ' '.join(tokens[left_context:index[0]]
            ) + ' ' + phrase + ' ' + ' '.join(tokens[index[-1] + 1:
            right_context])
        context_phrases.append(context_phrase.strip())
        if use_context_query:
            context_query = ' '.join(tokens[left_context:index[0]]
                ) + ' ' + query + ' ' + ' '.join(tokens[index[-1] + 1:
                right_context])
            context_queries.append(context_query.strip())
    context_phrase_embs = self.embed_batch(context_phrases, max_length=
        max_seq_length, contextual=True)
    phrase_embs = (self.
        extract_contextual_phrase_embeddings_with_context_window([phrase[0] for
        phrase in list_phrase], context_phrases, context_phrase_embs,
        max_length=max_seq_length))
    context_query_embs = None
    if use_context_query:
        context_query_embs = self.embed_batch(context_queries, max_length=
            max_seq_length, contextual=True)
        query_emb = (self.
            extract_contextual_phrase_embeddings_with_context_window([query
            ] * len(list_phrase), context_queries, context_query_embs,
            max_length=max_seq_length))
        score = cosine_similarity(query_emb, phrase_embs)
        del context_query_embs
        del query_emb
        del context_phrase_embs
        del phrase_embs
        torch.cuda.empty_cache()
        score = [float(score[i][i]) for i in range(len(score))]
        return score
    query_emb = self.embed_batch([query])
    score = cosine_similarity(query_emb, phrase_embs)
    del context_phrase_embs
    del phrase_embs
    torch.cuda.empty_cache()
    return score.tolist()[-1]
