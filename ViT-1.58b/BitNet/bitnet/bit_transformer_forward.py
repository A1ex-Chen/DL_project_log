def forward(self, x):
    x = self.emb(x)
    x = self.transformer(x)
    return self.to_logits(x)
