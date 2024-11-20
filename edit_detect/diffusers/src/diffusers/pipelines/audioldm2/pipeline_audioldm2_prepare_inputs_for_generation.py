def prepare_inputs_for_generation(inputs_embeds, attention_mask=None,
    past_key_values=None, **kwargs):
    if past_key_values is not None:
        inputs_embeds = inputs_embeds[:, -1:]
    return {'inputs_embeds': inputs_embeds, 'attention_mask':
        attention_mask, 'past_key_values': past_key_values, 'use_cache':
        kwargs.get('use_cache')}
