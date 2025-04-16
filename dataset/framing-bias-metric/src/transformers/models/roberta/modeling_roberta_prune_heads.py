def prune_heads(self, heads):
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(heads, self.self.
        num_attention_heads, self.self.attention_head_size, self.pruned_heads)
    self.self.query = prune_linear_layer(self.self.query, index)
    self.self.key = prune_linear_layer(self.self.key, index)
    self.self.value = prune_linear_layer(self.self.value, index)
    self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
    self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    self.self.all_head_size = (self.self.attention_head_size * self.self.
        num_attention_heads)
    self.pruned_heads = self.pruned_heads.union(heads)
