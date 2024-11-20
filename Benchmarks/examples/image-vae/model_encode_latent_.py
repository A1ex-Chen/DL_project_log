def encode_latent_(self, x1, x2):
    x1, x2 = self.encode(x1, x2)
    x = torch.cat([x1, x2], dim=1)
    mu, logvar = self.z_mean(x), self.z_log_var(x)
    z = self.reparam(logvar, mu)
    return z, mu, logvar
