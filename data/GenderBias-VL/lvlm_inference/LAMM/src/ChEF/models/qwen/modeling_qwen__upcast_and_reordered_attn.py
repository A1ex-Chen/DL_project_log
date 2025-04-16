def _upcast_and_reordered_attn(self, query, key, value,
    registered_causal_mask, attention_mask=None, head_mask=None):
    bsz, num_heads, q_seq_len, dk = query.size()
    _, _, k_seq_len, _ = key.size()
    attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype
        =torch.float32, device=query.device)
    scale_factor = 1.0
    if self.scale_attn_weights:
        scale_factor /= float(value.size(-1)) ** 0.5
    with autocast(enabled=False):
        q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
            -1, dk, k_seq_len)
        attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(),
            beta=0, alpha=scale_factor)
        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len,
            k_seq_len)
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = registered_causal_mask[:, :, key_length - query_length:
        key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
        attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    if attn_weights.dtype != torch.float32:
        raise RuntimeError(
            'Error with upcasting, attn_weights does not have dtype torch.float32'
            )
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights
