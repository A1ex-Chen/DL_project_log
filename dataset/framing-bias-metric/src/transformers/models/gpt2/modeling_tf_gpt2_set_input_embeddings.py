def set_input_embeddings(self, value):
    self.wte.weight = value
    self.wte.vocab_size = self.wte.weight.shape[0]
