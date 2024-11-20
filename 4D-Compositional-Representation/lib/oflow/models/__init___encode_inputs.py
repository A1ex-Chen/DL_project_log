def encode_inputs(self, inputs):
    """ Returns spatial and temporal latent code for inputs.

        Args:
            inputs (tensor): inputs tensor
        """
    c_s = self.encode_spatial_inputs(inputs)
    c_t = self.encode_temporal_inputs(inputs)
    return c_s, c_t
