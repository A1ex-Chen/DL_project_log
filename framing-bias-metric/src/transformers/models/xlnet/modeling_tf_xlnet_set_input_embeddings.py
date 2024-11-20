def set_input_embeddings(self, value):
    self.word_embedding.weight = value
    self.word_embedding.vocab_size = value.shape[0]
