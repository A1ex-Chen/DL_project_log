def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
    inputs_embeds=None, **kwargs):
    if inputs_embeds is not None:
        raise NotImplementedError(
            'inputs_embeds is not implemented for MPT yet')
    attention_mask = kwargs['attention_mask'].bool()
    if attention_mask[:, -1].sum() != attention_mask.shape[0]:
        raise NotImplementedError(
            'MPT does not support generation with right padding.')
    if self.transformer.attn_uses_sequence_id and self.training:
        sequence_id = torch.zeros_like(input_ids[:1])
    else:
        sequence_id = None
    if past_key_values is not None:
        input_ids = input_ids[:, -1].unsqueeze(-1)
    if self.transformer.prefix_lm:
        prefix_mask = torch.ones_like(attention_mask)
        if kwargs.get('use_cache') == False:
            raise NotImplementedError(
                'MPT with prefix_lm=True does not support use_cache=False.')
    else:
        prefix_mask = None
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
        'prefix_mask': prefix_mask, 'sequence_id': sequence_id,
        'past_key_values': past_key_values, 'use_cache': kwargs.get(
        'use_cache', True)}
