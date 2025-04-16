def prepare_inputs_for_generation(self, input_ids, past=None,
    attention_mask=None, use_cache=None, **kwargs):
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)
    if past:
        input_ids = input_ids[:, -1:]
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
        'past_key_values': past, 'use_cache': use_cache}
