def forward(self, t, p, T_batch=None, invert=False, **kwargs):
    """ Performs a forward pass through the network.

        Args:
            t (tensor): time values
            p (tensor): points
            T_batch (tensor): time helper tensor to perform batch processing
                when going backwards in time
            invert (bool): whether to go backwards
        """
    p, c, z = self.disentangle_inputs(p)
    p = self.concat_time_axis(t, p, T_batch, invert)
    net = self.fc_p(p)
    for i in range(self.n_blocks):
        if self.c_dim != 0:
            net_c = self.fc_c[i](c).unsqueeze(1)
            net = net + net_c
        if self.z_dim != 0:
            net_z = self.fc_z[i](z).unsqueeze(1)
            net = net + net_z
        net = self.blocks[i](net)
    motion_vectors = self.fc_out(self.actvn(net))
    sign = -1 if invert else 1
    motion_vectors = sign * motion_vectors
    out = self.concat_output(motion_vectors)
    return out
