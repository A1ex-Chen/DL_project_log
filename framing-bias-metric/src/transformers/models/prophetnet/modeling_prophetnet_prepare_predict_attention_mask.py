def prepare_predict_attention_mask(self, hidden_states, attention_mask):
    seq_length, batch_size = hidden_states.shape[:2]
    predict_causal_mask = ngram_attention_bias(self.max_target_positions,
        self.ngram, hidden_states.device, hidden_states.dtype)
    predict_causal_mask = torch.cat([predict_causal_mask[:, :seq_length, :
        seq_length], predict_causal_mask[:, :seq_length, self.
        max_target_positions:self.max_target_positions + seq_length]], dim=-1)
    extended_predict_causal_mask = predict_causal_mask[:, None, :, :].expand(
        predict_causal_mask.shape[:1] + (batch_size,) + predict_causal_mask
        .shape[1:])
    if attention_mask is not None:
        extended_attention_mask = (1.0 - attention_mask[None, :, None, :]
            ) * -10000.0
        extended_attention_mask = extended_attention_mask.expand((self.
            ngram, batch_size, seq_length, seq_length))
        extended_attention_mask = torch.cat([extended_attention_mask, torch
            .zeros_like(extended_attention_mask)], dim=-1)
        extended_predict_attention_mask = (extended_predict_causal_mask +
            extended_attention_mask)
    else:
        extended_predict_attention_mask = extended_predict_causal_mask
    return extended_predict_attention_mask.repeat(1, self.config.
        num_decoder_attention_heads, 1, 1).to(hidden_states.dtype)
