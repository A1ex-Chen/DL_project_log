def forward(self, x):
    mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    x = self.embedding(x)
    for transformer in self.transformer_blocks:
        x = transformer.forward(x, mask)
    return x
