def retrieve_indices(self, embeddings: np.ndarray, k: int):
    return self.index.retrieve_indices(embeddings, k)
