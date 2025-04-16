def get_input_dims(self) ->Tuple:
    """Returns dimension of input image

        Returns:
            `Tuple`: (height, width)
        """
    input_module = self.vqvae if self.vqvae is not None else self.unet
    sample_size = (input_module.sample_size, input_module.sample_size) if type(
        input_module.sample_size) == int else input_module.sample_size
    return sample_size
