def transform_to_t(self, t, p, z=None, c_t=None):
    """  Transforms points p from time 0 to (multiple) time values t.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        """
    p_out, _ = self.eval_ODE(t, p, c_t=c_t, z=z, return_start=0 in t)
    return p_out
