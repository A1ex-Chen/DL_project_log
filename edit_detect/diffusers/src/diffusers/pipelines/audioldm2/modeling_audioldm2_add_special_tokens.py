def add_special_tokens(hidden_states, attention_mask, sos_token, eos_token):
    batch_size = hidden_states.shape[0]
    if attention_mask is not None:
        new_attn_mask_step = attention_mask.new_ones((batch_size, 1))
        attention_mask = torch.concat([new_attn_mask_step, attention_mask,
            new_attn_mask_step], dim=-1)
    sos_token = sos_token.expand(batch_size, 1, -1)
    eos_token = eos_token.expand(batch_size, 1, -1)
    hidden_states = torch.concat([sos_token, hidden_states, eos_token], dim=1)
    return hidden_states, attention_mask
