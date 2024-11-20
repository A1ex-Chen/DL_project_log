def to_4d(self, attention_mask_2d: torch.Tensor, query_length: int,
    key_value_length: Optional[int]=None, dtype: torch.dtype=torch.float32
    ) ->torch.Tensor:
    """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
    input_shape = attention_mask_2d.shape[0], query_length
    causal_4d_mask = None
    if (input_shape[-1] > 1 or self.sliding_window is not None
        ) and self.is_causal:
        if key_value_length is None:
            raise ValueError(
                'This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask.'
                )
        past_key_values_length = key_value_length - query_length
        causal_4d_mask = self._make_causal_mask(input_shape, dtype, device=
            attention_mask_2d.device, past_key_values_length=
            past_key_values_length, sliding_window=self.sliding_window)
    elif self.sliding_window is not None:
        raise NotImplementedError(
            'Sliding window is currently only implemented for causal masking')
    expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype,
        tgt_len=input_shape[-1]).to(attention_mask_2d.device)
    expanded_4d_mask = (expanded_attn_mask if causal_4d_mask is None else 
        expanded_attn_mask + causal_4d_mask)
    return expanded_4d_mask
