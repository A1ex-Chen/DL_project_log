def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor
    ):
    s_k, s_q = attn_bias.shape[-2:]
    if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
        raise ValueError('attn_bias does not match the expected shape. ' +
            f'The last two dimensions should both be {self.config.max_length} '
             + f'but are {s_k} and {s_q}.')
    seq_len = prefix_mask.shape[-1]
    if seq_len > self.config.max_seq_len:
        raise ValueError(
            f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )
    attn_bias = attn_bias[..., :seq_len, :seq_len]
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool,
        device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
    prefix = prefix_mask.view(-1, 1, 1, seq_len)
    cannot_attend = ~torch.logical_or(causal, prefix.bool())
    min_val = torch.finfo(attn_bias.dtype).min
    attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
    return attn_bias
