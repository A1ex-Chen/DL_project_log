def forward(self, t, p, T_batch=None, invert=False, **kwargs):
    """ Performs a forward pass through the network.

        Args:
            t (tensor): time values
            p (tensor): points
            T_batch (tensor): time helper tensor to perform batch processing
                when going backwards in time
            invert (bool): whether to go backwards
        """
    p, c = self.disentangle_inputs(p)
    p = self.concat_time_axis(t, p, T_batch, invert)
    net = self.fc_p(p)
    for i in range(self.n_blocks):
        if self.c_dim != 0:
            net_c = self.fc_c[i](c)
            net = net + net_c
        net = self.blocks[i](net)
    motion_vectors = self.fc_out(self.actvn(net))
    sign = -1 if invert else 1
    motion_vectors = sign * motion_vectors
    out = self.concat_output(motion_vectors)
    return out
