def set_input_embeddings(self, new_embeddings):
    self.shared = new_embeddings
    self.encoder.set_input_embeddings(new_embeddings)
    self.decoder.set_input_embeddings(new_embeddings)
