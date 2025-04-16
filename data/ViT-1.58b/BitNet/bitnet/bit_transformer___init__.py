def __init__(self, dim: int, depth: int, num_tokens: int, heads=8, ff_mult=4):
    super().__init__()
    self.emb = nn.Embedding(num_tokens, dim)
    self.transformer = Transformer(dim=dim, depth=depth, heads=heads,
        ff_mult=ff_mult)
    self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))
