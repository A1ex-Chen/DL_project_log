def disentangle_vf_output(self, v_out, p_dim=3, c_dim=None, z_dim=None,
    return_start=False):
    """ Disentangles the output of the velocity field.

        The inputs and outputs for / of the velocity network are concatenated
        to be able to use the adjoint method.

        Args:
            v_out (tensor): output of the velocity field
            p_dim (int): points dimension
            c_dim (int): dimension of conditioned code c
            z_dim (int): dimension of latent code z
            return_start (bool): whether to return start points
        """
    n_steps, batch_size, _ = v_out.shape
    if z_dim is not None and z_dim != 0:
        v_out = v_out[:, :, :-z_dim]
    if c_dim is not None and c_dim != 0:
        v_out = v_out[:, :, :-c_dim]
    v_out = v_out.contiguous().view(n_steps, batch_size, -1, p_dim)
    if not return_start:
        v_out = v_out[1:]
    v_out = v_out.transpose(0, 1)
    return v_out
