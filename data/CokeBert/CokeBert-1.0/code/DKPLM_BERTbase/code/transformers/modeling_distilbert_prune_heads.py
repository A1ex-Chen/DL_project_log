def prune_heads(self, heads):
    attention_head_size = self.dim // self.n_heads
    if len(heads) == 0:
        return
    mask = torch.ones(self.n_heads, attention_head_size)
    heads = set(heads) - self.pruned_heads
    for head in heads:
        head -= sum(1 if h < head else 0 for h in self.pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    self.q_lin = prune_linear_layer(self.q_lin, index)
    self.k_lin = prune_linear_layer(self.k_lin, index)
    self.v_lin = prune_linear_layer(self.v_lin, index)
    self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
    self.n_heads = self.n_heads - len(heads)
    self.dim = attention_head_size * self.n_heads
    self.pruned_heads = self.pruned_heads.union(heads)
