def encode_inputs(self, inputs):
    """ Returns encoded latent code for inputs.

        Args:
            inputs (tensor): inputs tensor
        """
    c_p = self.encoder(inputs[:, 0, :])
    c_m = self.encoder_motion(inputs)
    c_i = self.encoder_identity(inputs[:, 0, :])
    return c_p, c_m, c_i
