def concat_output(self, out):
    """ Returns the output of the velocity network.

        The points, the conditioned codes c, and the latent codes z are
        concatenated to produce a single output of similar size as the input
        tensor. Zeros are concatenated with respect to the dimensions of the
        hidden vectors c and z. (This ensures that the "motion vectors" for
        these "fake points" are 0, and thus are not used by the adjoint method
        to calculate the step size.)

        Args:
            out (tensor): output points
        """
    batch_size = out.shape[0]
    device = out.device
    c_dim = self.c_dim
    out = out.contiguous().view(batch_size, -1)
    if c_dim != 0:
        c_out = torch.zeros(batch_size, c_dim).to(device)
        out = torch.cat([out, c_out], dim=-1)
    return out
