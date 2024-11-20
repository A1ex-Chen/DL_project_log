def forward(self, t, y):
    return torch.mm(y ** 3, self.A)
