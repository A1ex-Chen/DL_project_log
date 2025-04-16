def transform_to_t_backward(self, t, p, z=None, c_t=None):
    """ Transforms points p from time 1 (multiple) t backwards.

        For example, for t = [0.5, 1], it transforms the points from the
        coordinate system t = 1 to coordinate systems t = 0.5 and t = 0.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned code c
        """
    device = self.device
    batch_size = p.shape[0]
    p_out, _ = self.eval_ODE(t, p, c_t=c_t, z=z, t_batch=torch.ones(
        batch_size).to(device), invert=True, return_start=0 in t)
    return p_out
