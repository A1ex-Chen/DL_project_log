def _apply_sequence_id(self, attn_bias: torch.Tensor, sequence_id: torch.
    LongTensor):
    seq_len = sequence_id.shape[-1]
    if seq_len > self.config.max_seq_len:
        raise ValueError(
            f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )
    attn_bias = attn_bias[..., :seq_len, :seq_len]
    cannot_attend = torch.logical_not(torch.eq(sequence_id.view(-1, seq_len,
        1), sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
    min_val = torch.finfo(attn_bias.dtype).min
    attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
    return attn_bias
