def embed_tokens(self, token_ids):
    return self.llama_model.model.embed_tokens(token_ids)
