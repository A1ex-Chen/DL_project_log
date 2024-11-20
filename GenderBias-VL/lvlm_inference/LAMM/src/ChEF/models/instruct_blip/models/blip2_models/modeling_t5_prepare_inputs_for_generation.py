def prepare_inputs_for_generation(self, input_ids, past=None,
    attention_mask=None, head_mask=None, decoder_head_mask=None,
    cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
    if past is not None:
        input_ids = input_ids[:, -1:]
    return {'decoder_input_ids': input_ids, 'past_key_values': past,
        'encoder_outputs': encoder_outputs, 'attention_mask':
        attention_mask, 'head_mask': head_mask, 'decoder_head_mask':
        decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask,
        'use_cache': use_cache}
