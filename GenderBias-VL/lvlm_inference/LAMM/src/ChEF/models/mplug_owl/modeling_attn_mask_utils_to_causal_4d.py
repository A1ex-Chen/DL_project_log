def to_causal_4d(self, batch_size: int, query_length: int, key_value_length:
    int, dtype: torch.dtype=torch.float32, device: Union[torch.device,
    'str']='cpu') ->torch.Tensor:
    """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
    if not self.is_causal:
        raise ValueError(
            f'Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.'
            )
    input_shape = batch_size, query_length
    past_key_values_length = key_value_length - query_length
    causal_4d_mask = None
    if input_shape[-1] > 1 or self.sliding_window is not None:
        causal_4d_mask = self._make_causal_mask(input_shape, dtype, device=
            device, past_key_values_length=past_key_values_length,
            sliding_window=self.sliding_window)
    return causal_4d_mask
