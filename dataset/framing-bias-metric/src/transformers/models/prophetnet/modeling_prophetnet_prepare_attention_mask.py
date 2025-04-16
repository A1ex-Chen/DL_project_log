def prepare_attention_mask(self, hidden_states, attention_mask):
    seq_length, batch_size = hidden_states.shape[:2]
    causal_mask = hidden_states.new(seq_length, seq_length).float().fill_(-
        float('inf'))
    causal_mask = torch.triu(causal_mask, 1)
    extended_causal_mask = causal_mask[:seq_length, :seq_length][None, :, :
        ].expand((batch_size,) + causal_mask.shape)
    if attention_mask is not None:
        extended_attention_mask = (1.0 - attention_mask[:, None, :]) * -10000.0
        extended_attention_mask = (extended_causal_mask +
            extended_attention_mask)
    else:
        extended_attention_mask = extended_causal_mask
    return extended_attention_mask.repeat(self.config.
        num_decoder_attention_heads, 1, 1).to(hidden_states.dtype)
