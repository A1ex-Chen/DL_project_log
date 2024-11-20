def generate(self, x: Tensor, **kwargs) ->Tensor:
    """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
    return self.forward(x)[0]
