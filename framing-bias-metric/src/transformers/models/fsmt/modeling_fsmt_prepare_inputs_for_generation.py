def prepare_inputs_for_generation(self, decoder_input_ids, past=None,
    attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
    return {'input_ids': None, 'encoder_outputs': encoder_outputs,
        'past_key_values': past, 'decoder_input_ids': decoder_input_ids,
        'attention_mask': attention_mask, 'use_cache': use_cache}
