def set_output_embeddings(self, new_embeddings):
    self.cls.predictions.decoder = new_embeddings
