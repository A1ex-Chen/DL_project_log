def _get_embedding_shapes(self):
    embeddings = self.get_input_embeddings()
    return [emb.weight.shape[0] for emb in embeddings.emb_layers]
