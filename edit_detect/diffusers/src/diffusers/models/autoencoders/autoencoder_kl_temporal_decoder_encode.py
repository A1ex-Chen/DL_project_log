@apply_forward_hook
def encode(self, x: torch.Tensor, return_dict: bool=True) ->Union[
    AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
    """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
    h = self.encoder(x)
    moments = self.quant_conv(h)
    posterior = DiagonalGaussianDistribution(moments)
    if not return_dict:
        return posterior,
    return AutoencoderKLOutput(latent_dist=posterior)
