def forward(self, p, time_val, inputs, sample=True):
    """ Makes a forward pass through the network.

        Args:
            p (tensor): points tensor
            time_val (tensor): time values
            inputs (tensor): input tensor
            sample (bool): whether to sample
        """
    batch_size = p.size(0)
    c_s, c_t = self.encode_inputs(inputs)
    z, z_t = self.get_z_from_prior((batch_size,), sample=sample)
    p_t_at_t0 = self.model.transform_to_t0(time_val, p, c_t=c_t, z=z_t)
    out = self.model.decode(p_t_at_t0, c=c_s, z=z)
    return out
