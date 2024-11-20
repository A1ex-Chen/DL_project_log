def disentangle_inputs(self, inputs):
    """ Disentangles the inputs and returns the points and latent code
        tensors separately.

        The input consists of the latent code z, the latent conditioned code c,
        and the points. They are concatenated before using as input for the
        velocity field to be able to use the adjoint method to obtain
        gradients. Here, the full input tensor is disentangled again into its
        components.

        Args:
            inputs (tensor): velocity field inputs
        """
    c_dim = self.c_dim
    z_dim = self.z_dim
    batch_size, device = inputs.shape[0], inputs.device
    if z_dim is not None and z_dim != 0:
        z = inputs[:, -z_dim:]
        p = inputs[:, :-z_dim]
    else:
        z = torch.empty(batch_size, 0).to(device)
        p = inputs
    if c_dim is not None and c_dim != 0:
        c = p[:, -c_dim:]
        p = p[:, :-c_dim]
    else:
        c = torch.empty(batch_size, 0).to(device)
    p = p.view(batch_size, -1, self.out_dim)
    return p, c, z
