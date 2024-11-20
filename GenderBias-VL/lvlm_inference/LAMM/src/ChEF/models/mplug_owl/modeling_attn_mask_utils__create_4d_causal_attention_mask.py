def _create_4d_causal_attention_mask(input_shape: Union[torch.Size, Tuple,
    List], dtype: torch.dtype, device: torch.device, past_key_values_length:
    int=0, sliding_window: Optional[int]=None):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`

    Args:
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        device (`int`):
            The torch device the created mask shall have.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True,
        sliding_window=sliding_window)
    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(input_shape[0],
        input_shape[-1], key_value_length, dtype=dtype, device=device)
    return attention_mask
