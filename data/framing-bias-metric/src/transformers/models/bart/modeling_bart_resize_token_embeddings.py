def resize_token_embeddings(self, new_num_tokens: int) ->nn.Embedding:
    old_num_tokens = self.model.shared.num_embeddings
    new_embeddings = super().resize_token_embeddings(new_num_tokens)
    self.model.shared = new_embeddings
    self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
    return new_embeddings
