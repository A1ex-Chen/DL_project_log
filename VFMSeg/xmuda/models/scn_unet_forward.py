def forward(self, x):
    x = self.sparseModel(x)
    return x
