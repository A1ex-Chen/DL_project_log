def _value_per_attn_head(self, hidden_states):
    per_head_value = self.value.weight.reshape(self.num_attention_heads,
        self.attention_head_size, self.hidden_size).transpose(-2, -1)
    value_vectors = torch.einsum('balh,ahr->balr', hidden_states,
        per_head_value)
    return value_vectors
