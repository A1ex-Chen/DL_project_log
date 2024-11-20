def concat_vf_input(self, p, c=None):
    """ Concatenate points p and latent code c to use it as input for ODE Solver.

        p of size (B x T x dim) and c of size (B x c_dim) and z of size
        (B x z_dim) is concatenated to obtain a tensor of size
        (B x (T*dim) + c_dim + z_dim).

        This is done to be able to use to the adjont method for obtaining
        gradients.

        Args:
            p (tensor): points tensor
            c (tensor): latent conditioned code c
        """
    batch_size = p.shape[0]
    p_out = p.contiguous().view(batch_size, -1)
    if c is not None and c.shape[-1] != 0:
        assert c.shape[0] == batch_size
        c = c.contiguous().view(batch_size, -1)
        p_out = torch.cat([p_out, c], dim=-1)
    return p_out
