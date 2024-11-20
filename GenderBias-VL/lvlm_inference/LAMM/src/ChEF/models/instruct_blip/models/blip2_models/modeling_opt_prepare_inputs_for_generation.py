def prepare_inputs_for_generation(self, input_ids=None, query_embeds=None,
    past=None, attention_mask=None, use_cache=None, **kwargs):
    if attention_mask is None:
        if input_ids is not None:
            attention_mask = input_ids.new_ones(input_ids.shape)
    if past:
        input_ids = input_ids[:, -1:]
        query_embeds = None
    return {'input_ids': input_ids, 'query_embeds': query_embeds,
        'attention_mask': attention_mask, 'past_key_values': past,
        'use_cache': use_cache}
