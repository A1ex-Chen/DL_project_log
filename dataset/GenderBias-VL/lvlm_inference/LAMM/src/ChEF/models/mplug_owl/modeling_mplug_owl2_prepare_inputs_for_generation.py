def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
    attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}
    model_inputs.update({'past_key_values': past_key_values, 'use_cache':
        kwargs.get('use_cache'), 'attention_mask': attention_mask, 'images':
        kwargs.get('images', None)})
    return model_inputs
