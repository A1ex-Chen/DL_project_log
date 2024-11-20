def forward(self, x):
    x = self.forward_features(x)
    x = x.flatten(2).mean(-1)
    x = self.head(x)
    return x
