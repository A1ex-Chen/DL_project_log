def forward(self, x):
    return x[0].permute(2, 0, 1), *x[1:]
