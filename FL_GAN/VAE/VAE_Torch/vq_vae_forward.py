def forward(self, x):
    z_e = self.encoder(x)
    q, z_q = self.codebook(z_e)
    x_reconstructed = self.decoder(z_e)
    return x_reconstructed, z_e, z_q
