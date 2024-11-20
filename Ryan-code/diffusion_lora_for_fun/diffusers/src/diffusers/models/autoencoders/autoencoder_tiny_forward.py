def forward(self, sample: torch.Tensor, return_dict: bool=True) ->Union[
    DecoderOutput, Tuple[torch.Tensor]]:
    """
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
    enc = self.encode(sample).latents
    scaled_enc = self.scale_latents(enc).mul_(255).round_().byte()
    unscaled_enc = self.unscale_latents(scaled_enc / 255.0)
    dec = self.decode(unscaled_enc)
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
