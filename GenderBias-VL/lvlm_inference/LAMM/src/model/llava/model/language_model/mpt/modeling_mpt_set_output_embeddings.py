def set_output_embeddings(self, new_embeddings):
    self.transformer.wte = new_embeddings
