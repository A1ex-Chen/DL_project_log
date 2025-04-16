def _prepare_bart_decoder_inputs(self, inputs, decoder_input_ids=None,
    decoder_attn_mask=None, mask_dtype=None):
    """
        Prepare masks that ignore padding tokens decoder and a causal lm mask for the decoder if none are provided.
        This mimics the default behavior in fairseq. To override it pass in masks.
        """
    pad_token_id = self.config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = self._shift_right(inputs)
    bsz, tgt_len = shape_list(decoder_input_ids)[:2]
    if decoder_attn_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids,
            pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_attn_mask)
    causal_lm_mask = causal_attention_mask(tgt_len, tgt_len, mask_dtype)
    return decoder_input_ids, decoder_padding_mask, causal_lm_mask
