def forward(self, x):
    x1, x2 = x[0], x[1]
    return torch.add(x1, x2, alpha=self.a)
