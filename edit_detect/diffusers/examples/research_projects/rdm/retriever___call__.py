def __call__(self, embeddings, k: int=20):
    return self.index.retrieve_imgs(embeddings, k)
