def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
    attention_mask=None, inputs_embeds=None, im_mask=None, **kwargs):
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            remove_prefix_length = input_ids.shape[1] - 1
        input_ids = input_ids[:, remove_prefix_length:]
    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}
    im_mask = im_mask
    model_inputs.update({'position_ids': position_ids, 'past_key_values':
        past_key_values, 'use_cache': kwargs.get('use_cache'),
        'attention_mask': attention_mask, 'im_mask': im_mask})
    return model_inputs
