def _memory_efficient_attention_xformers(self, query, key, value,
    attention_mask):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key,
        value, attn_bias=attention_mask)
    return hidden_states
