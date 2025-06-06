@staticmethod
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype,
    device: torch.device, past_key_values_length: int=0, sliding_window:
    Optional[int]=None):
    """
        Make causal mask used for bi-directional self-attention.
        """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device
        )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length,
            dtype=dtype, device=device), mask], dim=-1)
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1
        context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int
            ), diagonal=diagonal)
        mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len +
        past_key_values_length)
