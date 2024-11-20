def set_input_embeddings(self, value):
    self.tokens_embed.weight = value
    self.tokens_embed.vocab_size = value.shape[0]
