def forward(self, x, emb):
    if self.act:
        emb = self.act(emb)
    emb = self.linear(emb)
    emb = emb[:, :, None, None]
    scale, shift = emb.chunk(2, dim=1)
    x = F.group_norm(x, self.num_groups, eps=self.eps)
    x = x * (1 + scale) + shift
    return x
