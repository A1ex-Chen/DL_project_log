def forward(self, sample: torch.FloatTensor, return_dict: bool=True) ->Union[
    DecoderOutput, torch.FloatTensor]:
    """
        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
    x = sample
    h = self.encode(x).latents
    dec = self.decode(h).sample
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
