def get_rel_embedding(self):
    rel_embeddings = (self.rel_embeddings.weight if self.relative_attention
         else None)
    return rel_embeddings
