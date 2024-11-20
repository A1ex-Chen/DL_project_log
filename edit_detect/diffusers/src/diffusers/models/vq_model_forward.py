def forward(self, sample: torch.Tensor, return_dict: bool=True) ->Union[
    DecoderOutput, Tuple[torch.Tensor, ...]]:
    """
        The [`VQModel`] forward method.

        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vq_model.VQEncoderOutput`] is returned, otherwise a plain `tuple`
                is returned.
        """
    h = self.encode(sample).latents
    dec = self.decode(h).sample
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
