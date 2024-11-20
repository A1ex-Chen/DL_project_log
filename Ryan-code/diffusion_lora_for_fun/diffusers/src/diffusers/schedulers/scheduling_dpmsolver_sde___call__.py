def __call__(self, sigma, sigma_next):
    t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.
        as_tensor(sigma_next))
    return self.tree(t0, t1) / (t1 - t0).abs().sqrt()
