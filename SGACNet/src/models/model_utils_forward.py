def forward(self, x):
    return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0
