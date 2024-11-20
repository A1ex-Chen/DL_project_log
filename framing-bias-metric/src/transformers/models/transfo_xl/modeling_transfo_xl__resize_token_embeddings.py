def _resize_token_embeddings(self, new_num_tokens, layer=-1):
    embeddings = self.get_input_embeddings()
    if new_num_tokens is None:
        return embeddings
    new_embeddings_layer = self._get_resized_embeddings(embeddings.
        emb_layers[layer], new_num_tokens)
    embeddings.emb_layers[layer] = new_embeddings_layer
    self.set_input_embeddings(embeddings)
    return self.get_input_embeddings()
