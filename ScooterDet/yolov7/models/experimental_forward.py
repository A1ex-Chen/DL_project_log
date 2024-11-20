def forward(self, x):
    x = self.model(x)
    x = self.end2end(x)
    return x
