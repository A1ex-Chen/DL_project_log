def _bottleneck(self, h):
    mu, logvar = self.fc1(h), self.fc2(h)
    z = self._reparameterize(mu, logvar)
    return z, mu, logvar
