def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
    inputs_embeds, past_key_values_length):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        if self.bidirectional_mask == 'g':
            bsz, src_length = input_shape
            combined_attention_mask = torch.zeros((bsz, 1, src_length, 
                src_length + past_key_values_length), dtype=inputs_embeds.
                dtype, device=inputs_embeds.device)
        else:
            combined_attention_mask = _make_causal_mask_opt(input_shape,
                inputs_embeds.dtype, past_key_values_length=
                past_key_values_length).to(inputs_embeds.device)
            if self.bidirectional_mask is not None:
                assert attention_mask.shape == self.bidirectional_mask.shape
                expanded_bidirectional_mask = _expand_mask_opt(self.
                    bidirectional_mask, inputs_embeds.dtype, tgt_len=
                    input_shape[-1]).to(inputs_embeds.device)
                combined_attention_mask = torch.maximum(
                    expanded_bidirectional_mask, combined_attention_mask)
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask_opt(attention_mask, inputs_embeds
            .dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = (expanded_attn_mask if 
            combined_attention_mask is None else expanded_attn_mask +
            combined_attention_mask)
    return combined_attention_mask
