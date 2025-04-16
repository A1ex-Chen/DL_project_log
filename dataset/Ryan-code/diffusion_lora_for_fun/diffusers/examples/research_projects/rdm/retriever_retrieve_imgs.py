def retrieve_imgs(self, embeddings: np.ndarray, k: int):
    return self.index.retrieve_imgs(embeddings, k)
