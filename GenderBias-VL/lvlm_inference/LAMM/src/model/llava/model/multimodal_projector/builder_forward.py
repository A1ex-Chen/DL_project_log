def forward(self, x):
    x = self.pre_norm(x)
    return x + self.proj(x)
