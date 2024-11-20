def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
    inputs_embeds=None, **kwargs):
    images = kwargs.pop('images', None)
    _inputs = super().prepare_inputs_for_generation(input_ids,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
    if images is not None:
        _inputs['images'] = images
    return _inputs
