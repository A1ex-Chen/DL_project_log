@torch.no_grad()
def encode_with_pretrained(self, x):
    c = self.pretrained_model.encode(x)
    if isinstance(c, DiagonalGaussianDistribution):
        c = c.mode()
    return c
