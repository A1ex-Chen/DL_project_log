def prune_heads(self, heads):
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(heads, self.
        num_attention_heads, self.attention_head_size, self.pruned_heads)
    self.query = prune_linear_layer(self.query, index)
    self.key = prune_linear_layer(self.key, index)
    self.value = prune_linear_layer(self.value, index)
    self.dense = prune_linear_layer(self.dense, index, dim=1)
    self.num_attention_heads = self.num_attention_heads - len(heads)
    self.all_head_size = self.attention_head_size * self.num_attention_heads
    self.pruned_heads = self.pruned_heads.union(heads)
