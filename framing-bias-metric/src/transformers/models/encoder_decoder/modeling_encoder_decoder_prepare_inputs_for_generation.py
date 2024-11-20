def prepare_inputs_for_generation(self, input_ids, past=None,
    attention_mask=None, encoder_outputs=None, **kwargs):
    decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)
    decoder_attention_mask = decoder_inputs['attention_mask'
        ] if 'attention_mask' in decoder_inputs else None
    input_dict = {'attention_mask': attention_mask,
        'decoder_attention_mask': decoder_attention_mask,
        'decoder_input_ids': decoder_inputs['input_ids'], 'encoder_outputs':
        encoder_outputs}
    if 'use_cache' in decoder_inputs:
        input_dict['decoder_use_cache'] = decoder_inputs['use_cache']
    if 'past_key_values' in decoder_inputs:
        input_dict['past_key_values'] = decoder_inputs['past_key_values']
    return input_dict
