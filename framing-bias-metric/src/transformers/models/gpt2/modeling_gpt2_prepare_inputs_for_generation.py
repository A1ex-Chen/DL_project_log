def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get('token_type_ids', None)
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
    attention_mask = kwargs.get('attention_mask', None)
    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None
    return {'input_ids': input_ids, 'past_key_values': past, 'use_cache':
        kwargs.get('use_cache'), 'position_ids': position_ids,
        'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
