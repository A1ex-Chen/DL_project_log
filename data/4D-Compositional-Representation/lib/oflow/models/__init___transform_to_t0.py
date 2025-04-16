def transform_to_t0(self, t, p, z=None, c_t=None):
    """ Transforms the points p at time t to time 0.

        Args:
            t (tensor): time values of the points
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
        """
    p_out, t_order = self.eval_ODE(t, p, c_t=c_t, z=z, t_batch=t, invert=
        True, return_start=True)
    batch_size = len(t_order)
    p_out = p_out[torch.arange(batch_size), t_order]
    return p_out
