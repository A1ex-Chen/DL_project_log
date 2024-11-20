def prepare_inputs_for_generation(self, input_ids, past=None,
    attention_mask=None, **model_kwargs):
    input_shape = input_ids.shape
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_shape)
    if past is not None:
        input_ids = input_ids[:, -1:]
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
        'past_key_values': past, 'encoder_hidden_states': model_kwargs.get(
        'encoder_hidden_states', None), 'encoder_attention_mask':
        model_kwargs.get('encoder_attention_mask', None), 'is_decoder': True}
