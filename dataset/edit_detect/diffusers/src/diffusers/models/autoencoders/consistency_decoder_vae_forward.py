def forward(self, sample: torch.Tensor, sample_posterior: bool=False,
    return_dict: bool=True, generator: Optional[torch.Generator]=None) ->Union[
    DecoderOutput, Tuple[torch.Tensor]]:
    """
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*, defaults to `None`):
                Generator to use for sampling.

        Returns:
            [`DecoderOutput`] or `tuple`:
                If return_dict is True, a [`DecoderOutput`] is returned, otherwise a plain `tuple` is returned.
        """
    x = sample
    posterior = self.encode(x).latent_dist
    if sample_posterior:
        z = posterior.sample(generator=generator)
    else:
        z = posterior.mode()
    dec = self.decode(z, generator=generator).sample
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
