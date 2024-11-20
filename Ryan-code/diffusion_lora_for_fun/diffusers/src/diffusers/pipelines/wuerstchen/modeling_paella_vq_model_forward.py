def forward(self, sample: torch.Tensor, return_dict: bool=True) ->Union[
    DecoderOutput, torch.Tensor]:
    """
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
    x = sample
    h = self.encode(x).latents
    dec = self.decode(h).sample
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
