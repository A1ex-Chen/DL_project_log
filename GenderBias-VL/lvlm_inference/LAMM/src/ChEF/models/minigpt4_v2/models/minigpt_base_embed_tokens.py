def embed_tokens(self, token_ids):
    if hasattr(self.llama_model.base_model, 'model'):
        embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids
            )
    else:
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
    return embeds
