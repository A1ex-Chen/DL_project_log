def decode(self, p, c=None, **kwargs):
    """ Returns occupancy values for the points p at time step t.

        Args:
            p (tensor): points of dimension 4
            c (tensor): latent conditioned code c (For OFlow, this is
                c_spatial, whereas for ONet 4D, this is c_temporal)
        """
    logits = self.decoder(p, c, **kwargs)
    p_r = dist.Bernoulli(logits=logits)
    return p_r
