def _build_causal_attention_mask(self, bsz, seq_len, dtype):
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)
    mask = mask.unsqueeze(1)
    return mask
