def retrieve_imgs_batch(self, embeddings: np.ndarray, k: int):
    return self.index.retrieve_imgs_batch(embeddings, k)
