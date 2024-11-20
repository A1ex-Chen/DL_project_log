def set_input_embeddings(self, value):
    self.w.weight = value
    self.w.vocab_size = value.shape[0]
