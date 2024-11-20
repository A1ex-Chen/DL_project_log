def prune_heads(self, heads):
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(heads, self.attention.
        num_attention_heads, self.attention.attention_head_size, self.
        pruned_heads)
    self.attention.query = prune_linear_layer(self.attention.query, index)
    self.attention.key = prune_linear_layer(self.attention.key, index)
    self.attention.value = prune_linear_layer(self.attention.value, index)
    self.output.dense = prune_linear_layer(self.output.out_proj, index, dim=1)
    self.attention.num_attention_heads = (self.attention.
        num_attention_heads - len(heads))
    self.attention.all_head_size = (self.attention.attention_head_size *
        self.attention.num_attention_heads)
    self.pruned_heads = self.pruned_heads.union(heads)
