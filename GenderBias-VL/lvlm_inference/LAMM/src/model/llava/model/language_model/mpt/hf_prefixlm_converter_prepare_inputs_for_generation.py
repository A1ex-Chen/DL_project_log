def prepare_inputs_for_generation(self: BloomForCausalLM, input_ids: torch.
    LongTensor, past: Optional[torch.Tensor]=None, attention_mask: Optional
    [torch.Tensor]=None, **kwargs) ->dict:
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        bidirectional_mask = None
        if past[0][0].shape[0] == input_ids.shape[0]:
            past = self._convert_to_bloom_cache(past)
    else:
        bidirectional_mask = torch.ones_like(input_ids)
    return {'input_ids': input_ids, 'past_key_values': past, 'use_cache': 
        True, 'attention_mask': attention_mask, 'bidirectional_mask':
        bidirectional_mask}
