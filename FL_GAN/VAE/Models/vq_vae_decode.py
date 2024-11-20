def decode(self, z: Tensor) ->Tensor:
    """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
    result = self.decoder(z)
    return result
