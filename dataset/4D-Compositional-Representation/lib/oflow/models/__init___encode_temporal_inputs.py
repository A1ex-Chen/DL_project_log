def encode_temporal_inputs(self, inputs):
    """ Returns the temporal encoding c_t.

        Args:
            inputs (tensor): input tensor)
        """
    batch_size = inputs.shape[0]
    device = self.device
    """
        if self.input_type == 'idx':
            c_t = self.encoder(inputs)
        """
    if self.encoder_temporal is not None:
        c_t = self.encoder_temporal(inputs)
    else:
        c_t = torch.empty(batch_size, 0).to(device)
    return c_t
