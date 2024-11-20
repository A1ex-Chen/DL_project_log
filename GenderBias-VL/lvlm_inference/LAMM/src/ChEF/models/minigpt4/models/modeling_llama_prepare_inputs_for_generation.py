def prepare_inputs_for_generation(self, input_ids, query_embeds=None,
    past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)
            query_embeds = None
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}
    model_inputs.update({'position_ids': position_ids, 'query_embeds':
        query_embeds, 'past_key_values': past_key_values, 'use_cache':
        kwargs.get('use_cache'), 'attention_mask': attention_mask})
    return model_inputs
