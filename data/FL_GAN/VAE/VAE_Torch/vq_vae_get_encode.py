def get_encode(self, x):
    z_e = self.encoder(x)
    q, z_q = self.codebook(z_e)
    return q
