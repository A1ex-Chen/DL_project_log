def encode(self, input: Tensor) ->List[Tensor]:
    """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
    result = self.encoder(input)
    return [result]
