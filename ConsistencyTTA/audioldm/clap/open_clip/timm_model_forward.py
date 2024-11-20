def forward(self, x):
    x = self.trunk(x)
    x = self.head(x)
    return x
