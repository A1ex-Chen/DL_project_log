def encode_spatial_inputs(self, inputs):
    """ Returns the spatial encoding c_s

        Args:
            inputs (tensor): inputs tensor
        """
    batch_size = inputs.shape[0]
    device = self.device
    if len(inputs.shape) > 1:
        inputs = inputs[:, 0, :]
    if self.encoder is not None:
        c = self.encoder(inputs)
    else:
        c = torch.empty(batch_size, 0).to(device)
    return c
