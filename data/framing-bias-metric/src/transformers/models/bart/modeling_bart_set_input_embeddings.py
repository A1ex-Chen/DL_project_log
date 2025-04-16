def set_input_embeddings(self, value):
    self.shared = value
    self.encoder.embed_tokens = self.shared
    self.decoder.embed_tokens = self.shared
