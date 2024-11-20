def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) ->Tuple[
    np.ndarray, np.ndarray]:
    _, ids = self.dataset.search_batch('embeddings', question_hidden_states,
        n_docs)
    docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
    vectors = [doc['embeddings'] for doc in docs]
    for i in range(len(vectors)):
        if len(vectors[i]) < n_docs:
            vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(
                vectors[i]), self.vector_size))])
    return np.array(ids), np.array(vectors)
