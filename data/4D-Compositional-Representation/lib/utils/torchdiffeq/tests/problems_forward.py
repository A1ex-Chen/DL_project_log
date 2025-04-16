def forward(self, t, y):
    return torch.mm(self.A, y.reshape(self.dim, 1)).reshape(-1)
