def forward(self, x, conditioning_emb):
    emb = self.scale_bias(conditioning_emb)
    scale, shift = torch.chunk(emb, 2, -1)
    x = x * (1 + scale) + shift
    return x
