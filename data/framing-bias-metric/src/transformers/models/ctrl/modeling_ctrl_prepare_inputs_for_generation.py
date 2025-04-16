def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=
    None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
    return {'input_ids': input_ids, 'past_key_values': past, 'use_cache':
        use_cache}
