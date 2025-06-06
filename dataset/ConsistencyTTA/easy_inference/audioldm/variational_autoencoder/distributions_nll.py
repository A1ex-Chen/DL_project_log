def nll(self, sample, dims=[1, 2, 3]):
    if self.deterministic:
        return torch.Tensor([0.0])
    logtwopi = np.log(2.0 * np.pi)
    return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self
        .mean, 2) / self.var, dim=dims)
