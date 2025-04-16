def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
    inputs_embeds, past_key_values_length):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(input_shape,
            inputs_embeds.dtype, device=inputs_embeds.device,
            past_key_values_length=past_key_values_length)
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.
            dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = (expanded_attn_mask if 
            combined_attention_mask is None else expanded_attn_mask +
            combined_attention_mask)
    return combined_attention_mask
