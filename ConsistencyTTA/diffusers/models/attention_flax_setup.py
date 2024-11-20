def setup(self):
    inner_dim = self.dim * 4
    self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)
