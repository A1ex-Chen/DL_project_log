def _resize_token_embeddings(self, new_num_tokens):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = self._get_resized_embeddings(old_embeddings,
        new_num_tokens)
    self.set_input_embeddings(new_embeddings)
    return self.get_input_embeddings()
