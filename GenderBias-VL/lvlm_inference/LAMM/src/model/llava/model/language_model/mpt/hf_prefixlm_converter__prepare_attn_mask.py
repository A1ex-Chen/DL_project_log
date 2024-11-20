def _prepare_attn_mask(self: BloomModel, attention_mask: torch.Tensor,
    bidirectional_mask: Optional[torch.Tensor], input_shape: Tuple[int, int
    ], past_key_values_length: int) ->torch.BoolTensor:
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape
    if src_length > 1:
        combined_attention_mask = _make_causal_mask_bloom(input_shape,
            device=device, past_key_values_length=past_key_values_length)
        if bidirectional_mask is not None:
            assert attention_mask.shape == bidirectional_mask.shape
            expanded_bidirectional_mask = _expand_mask_bloom(bidirectional_mask
                , tgt_length=src_length)
            combined_attention_mask = torch.logical_and(combined_attention_mask
                , expanded_bidirectional_mask)
    expanded_attn_mask = _expand_mask_bloom(attention_mask, tgt_length=
        src_length)
    combined_attention_mask = (expanded_attn_mask if 
        combined_attention_mask is None else expanded_attn_mask |
        combined_attention_mask)
    return combined_attention_mask
