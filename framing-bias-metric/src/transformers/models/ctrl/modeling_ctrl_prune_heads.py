def prune_heads(self, heads):
    attention_head_size = self.d_model_size // self.num_heads
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(heads, self.num_heads,
        attention_head_size, self.pruned_heads)
    self.Wq = prune_linear_layer(self.Wq, index)
    self.Wk = prune_linear_layer(self.Wk, index)
    self.Wv = prune_linear_layer(self.Wv, index)
    self.dense = prune_linear_layer(self.dense, index, dim=1)
    self.num_heads = self.num_heads - len(heads)
    self.d_model_size = attention_head_size * self.num_heads
    self.pruned_heads = self.pruned_heads.union(heads)
