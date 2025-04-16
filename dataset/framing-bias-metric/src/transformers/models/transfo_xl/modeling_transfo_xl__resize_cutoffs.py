def _resize_cutoffs(self, new_num_tokens, new_emb_size,
    new_embedding_shapes, layer):
    new_cutoffs = super()._resize_cutoffs(new_num_tokens, new_emb_size,
        new_embedding_shapes, layer)
    self.crit.cutoffs = new_cutoffs
    self.crit.cutoff_ends = [0] + new_cutoffs
    self.crit.n_token = new_num_tokens
