def encode_rgb(self, rgb_in: torch.Tensor) ->torch.Tensor:
    """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
    h = self.vae.encoder(rgb_in)
    moments = self.vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    rgb_latent = mean * self.rgb_latent_scale_factor
    return rgb_latent
