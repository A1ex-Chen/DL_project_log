def retrieve_indices_batch(self, embeddings: np.ndarray, k: int):
    return self.index.retrieve_indices_batch(embeddings, k)
