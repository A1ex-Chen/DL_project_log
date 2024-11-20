def _reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    esp = torch.randn_like(std)
    z = mu + std * esp
    return z
