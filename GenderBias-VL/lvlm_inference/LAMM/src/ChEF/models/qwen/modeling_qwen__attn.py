def _attn(self, query, key, value, registered_causal_mask, attention_mask=
    None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full([], value.size(-1) ** 0.5,
            dtype=attn_weights.dtype, device=attn_weights.device)
    query_length, key_length = query.size(-2), key.size(-2)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    return attn_output, attn_weights
