def prepare_inputs_for_generation(self, inputs, past, attention_mask,
    use_cache, **kwargs):
    assert past is not None, 'past has to be defined for encoder_outputs'
    if len(past) < 2:
        encoder_outputs, past_key_values = past, None
    else:
        encoder_outputs, past_key_values = past[0], past[1]
    if past_key_values is not None:
        inputs = inputs[:, -1:]
    return {'input_ids': None, 'decoder_input_ids': inputs,
        'past_key_values': past_key_values, 'encoder_outputs':
        encoder_outputs, 'attention_mask': attention_mask, 'use_cache':
        use_cache}
