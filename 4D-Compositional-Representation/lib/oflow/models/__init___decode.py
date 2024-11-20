def decode(self, p, z=None, c=None, **kwargs):
    """ Returns occupancy values for the points p at time step 0.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c (For OFlow, this is
                c_spatial)
        """
    logits = self.decoder(p, z, c, **kwargs)
    p_r = dist.Bernoulli(logits=logits)
    return p_r
