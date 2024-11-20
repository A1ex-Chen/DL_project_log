def prepare_inputs_for_generation(self, input_ids, query_embeds, past=None,
    attention_mask=None, **model_kwargs):
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)
    query_mask = input_ids.new_ones(query_embeds.shape[:-1])
    attention_mask = torch.cat([query_mask, attention_mask], dim=-1)
    if past is not None:
        input_ids = input_ids[:, -1:]
    return {'input_ids': input_ids, 'query_embeds': query_embeds,
        'attention_mask': attention_mask, 'past_key_values': past,
        'encoder_hidden_states': model_kwargs.get('encoder_hidden_states',
        None), 'encoder_attention_mask': model_kwargs.get(
        'encoder_attention_mask', None), 'is_decoder': True}
