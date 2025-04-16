def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype,
    past_key_values_length: int=0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length,
            dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len +
        past_key_values_length)
