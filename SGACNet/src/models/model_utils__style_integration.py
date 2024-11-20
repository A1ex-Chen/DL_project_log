def _style_integration(self, t):
    z = t * self.cfc[None, :, :]
    z = torch.sum(z, dim=2)[:, :, None, None]
    z_hat = self.bn(z)
    g = self.activation(z_hat)
    return g
