def __init__(self, parameters, deterministic=False):
    self.parameters = parameters
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)
    if self.deterministic:
        self.var = self.std = torch.zeros_like(self.mean, device=self.
            parameters.device, dtype=self.parameters.dtype)
