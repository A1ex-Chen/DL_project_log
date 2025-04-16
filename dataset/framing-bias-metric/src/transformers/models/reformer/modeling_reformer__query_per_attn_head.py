def _query_per_attn_head(self, hidden_states):
    per_head_query_key = self.query_key.weight.reshape(self.
        num_attention_heads, self.attention_head_size, self.hidden_size
        ).transpose(-2, -1)
    query_key_vectors = torch.einsum('balh,ahr->balr', hidden_states,
        per_head_query_key)
    return query_key_vectors
