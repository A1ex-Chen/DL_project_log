def __call__(self, module, step=None):
    if step is None:
        mu = self.mu
    else:
        mu = min(self.mu, (1.0 + step) / (10 + step))
    for name, x in module.state_dict().items():
        if name in self.shadow:
            new_average = (1.0 - mu) * x + mu * self.shadow[name]
            self.shadow[name] = new_average.clone()
        else:
            self.shadow[name] = x.clone()
