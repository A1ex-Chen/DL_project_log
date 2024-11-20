def invert_attention_mask(self, encoder_attention_mask: Tensor) ->Tensor:
    """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None,
            None, :]
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype
        =self.dtype)
    if self.dtype == torch.float16:
        encoder_extended_attention_mask = (1.0 -
            encoder_extended_attention_mask) * -10000.0
    elif self.dtype == torch.float32:
        encoder_extended_attention_mask = (1.0 -
            encoder_extended_attention_mask) * -1000000000.0
    else:
        raise ValueError(
            '{} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`'
            .format(self.dtype))
    return encoder_extended_attention_mask
