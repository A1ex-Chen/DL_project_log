def prune_heads(self, heads):
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(heads, self.n_heads,
        self.d_kv, self.pruned_heads)
    self.q = prune_linear_layer(self.q, index)
    self.k = prune_linear_layer(self.k, index)
    self.v = prune_linear_layer(self.v, index)
    self.o = prune_linear_layer(self.o, index, dim=1)
    self.n_heads = self.n_heads - len(heads)
    self.inner_dim = self.d_kv * self.n_heads
    self.pruned_heads = self.pruned_heads.union(heads)
