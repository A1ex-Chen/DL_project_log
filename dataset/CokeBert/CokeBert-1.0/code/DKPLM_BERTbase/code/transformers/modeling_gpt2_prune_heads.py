def prune_heads(self, heads):
    if len(heads) == 0:
        return
    mask = torch.ones(self.n_head, self.split_size // self.n_head)
    heads = set(heads) - self.pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    index_attn = torch.cat([index, index + self.split_size, index + 2 *
        self.split_size])
    self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
    self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
    self.split_size = self.split_size // self.n_head * (self.n_head - len(
        heads))
    self.n_head = self.n_head - len(heads)
    self.pruned_heads = self.pruned_heads.union(heads)
