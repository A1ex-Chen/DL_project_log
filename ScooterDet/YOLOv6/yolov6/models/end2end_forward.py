def forward(self, x):
    if self.with_preprocess:
        x = x[:, [2, 1, 0], ...]
        x = x * (1 / 255)
    x = self.model(x)
    if isinstance(x, list):
        x = x[0]
    else:
        x = x
    x = self.end2end(x)
    return x
