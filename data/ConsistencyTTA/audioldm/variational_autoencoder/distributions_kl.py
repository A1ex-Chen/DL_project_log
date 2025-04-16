def kl(self, other=None):
    if self.deterministic:
        return torch.Tensor([0.0])
    elif other is None:
        return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 -
            self.logvar, dim=[1, 2, 3])
    else:
        return 0.5 * torch.mean(torch.pow(self.mean - other.mean, 2) /
            other.var + self.var / other.var - 1.0 - self.logvar + other.
            logvar, dim=[1, 2, 3])
