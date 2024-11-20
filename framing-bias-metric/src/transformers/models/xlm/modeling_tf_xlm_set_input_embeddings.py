def set_input_embeddings(self, value):
    self.embeddings.weight = value
    self.embeddings.vocab_size = value.shape[0]
