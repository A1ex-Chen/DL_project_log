def _relative_shift_gather(positional_attn, context_len, shift):
    batch_size, n_head, seq_len, max_rel_len = positional_attn.shape
    positional_attn = torch.reshape(positional_attn, [batch_size, n_head,
        max_rel_len, seq_len])
    positional_attn = positional_attn[:, :, shift:, :]
    positional_attn = torch.reshape(positional_attn, [batch_size, n_head,
        seq_len, max_rel_len - shift])
    positional_attn = positional_attn[..., :context_len]
    return positional_attn
