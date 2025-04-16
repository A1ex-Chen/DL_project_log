def forward(self, x):
    for block in self.blocks:
        x = block(x) + x
        x = self.gelu(x)
    x = self.tail(x)
    return x
