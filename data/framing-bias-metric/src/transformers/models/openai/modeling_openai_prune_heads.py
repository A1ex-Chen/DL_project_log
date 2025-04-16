def prune_heads(self, heads):
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(heads, self.n_head, 
        self.split_size // self.n_head, self.pruned_heads)
    index_attn = torch.cat([index, index + self.split_size, index + 2 *
        self.split_size])
    self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
    self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
    self.split_size = self.split_size // self.n_head * (self.n_head - len(
        heads))
    self.n_head = self.n_head - len(heads)
    self.pruned_heads = self.pruned_heads.union(heads)
