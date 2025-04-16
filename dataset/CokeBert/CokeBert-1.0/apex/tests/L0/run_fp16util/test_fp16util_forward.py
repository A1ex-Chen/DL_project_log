def forward(self, x):
    return self.dn(self.bn(x))
