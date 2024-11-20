def forward(self, x):
    x = self.forward_features(x)
    return x
