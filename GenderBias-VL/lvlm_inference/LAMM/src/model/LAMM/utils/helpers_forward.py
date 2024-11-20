def forward(self, x, seq_len):
    assert x.ndim == 3
    x = x[torch.arange(x.shape[0]), seq_len]
    x = self.proj(x)
    return x
