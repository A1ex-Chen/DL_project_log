def transform_to_t(self, t, p, c_t=None):
    """  Transforms points p from time 0 to (multiple) time values t.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        """
    p_out, t_order = self.eval_ODE(t, p, c_t=c_t, return_start=True)
    batch_size = len(t_order)
    p_out = p_out[torch.arange(batch_size), t_order]
    return p_out
