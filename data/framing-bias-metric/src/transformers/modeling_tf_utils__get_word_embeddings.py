def _get_word_embeddings(self, embeddings):
    if hasattr(embeddings, 'word_embeddings'):
        return embeddings.word_embeddings
    elif hasattr(embeddings, 'weight'):
        return embeddings.weight
    else:
        raise ValueError('word embedding is not defined.')
