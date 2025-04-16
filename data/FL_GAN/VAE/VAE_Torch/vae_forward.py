def forward(self, x):
    z_mu, z_log_sigma = torch.chunk(self.encoder(x), 2, dim=1)
    z_sigma = torch.exp(z_log_sigma)
    z = self.reparameterize(z_mu, z_sigma)
    x_mu = self.decoder(z)
    x_sigma = torch.ones_like(x_mu)
    return x_mu, x_sigma, z, z_mu, z_sigma
