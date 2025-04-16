def scaled_dot_product_attention(q, k, v, mask, attention_mask=None,
    head_mask=None):
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -10000.0
    if attention_mask is not None:
        scaled_attention_logits = scaled_attention_logits + attention_mask
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    output = torch.matmul(attention_weights, v)
    return output, attention_weights
