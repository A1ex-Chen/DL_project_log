def set_input_embeddings(self, value):
    self.embeddings.word_embeddings = value
    self.embeddings.vocab_size = value.shape[0]
