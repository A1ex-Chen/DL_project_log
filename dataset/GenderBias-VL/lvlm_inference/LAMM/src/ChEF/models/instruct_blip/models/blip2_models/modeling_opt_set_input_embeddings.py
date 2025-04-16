def set_input_embeddings(self, value):
    self.model.decoder.embed_tokens = value
