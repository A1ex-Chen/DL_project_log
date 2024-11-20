def forward(self, sample: torch.FloatTensor, sample_posterior: bool=False,
    return_dict: bool=True, generator: Optional[torch.Generator]=None) ->Union[
    DecoderOutput, torch.FloatTensor]:
    """
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
    x = sample
    posterior = self.encode(x).latent_dist
    if sample_posterior:
        z = posterior.sample(generator=generator)
    else:
        z = posterior.mode()
    dec = self.decode(z).sample
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
