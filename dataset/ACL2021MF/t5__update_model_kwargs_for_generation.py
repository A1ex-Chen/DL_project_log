@staticmethod
def _update_model_kwargs_for_generation(outputs: ModelOutput, model_kwargs:
    Dict[str, Any], is_encoder_decoder: bool=False) ->Dict[str, Any]:
    if 'past_key_values' in outputs:
        model_kwargs['past'] = outputs.past_key_values
    elif 'mems' in outputs:
        model_kwargs['past'] = outputs.mems
    elif 'past_buckets_states' in outputs:
        model_kwargs['past'] = outputs.past_buckets_states
    else:
        model_kwargs['past'] = None
    model_kwargs['decoder_mention_flag'] = outputs.decoder_mention_flags
    if 'token_type_ids' in model_kwargs:
        token_type_ids = model_kwargs['token_type_ids']
        model_kwargs['token_type_ids'] = torch.cat([token_type_ids,
            token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
    if not is_encoder_decoder:
        if 'attention_mask' in model_kwargs:
            attention_mask = model_kwargs['attention_mask']
            model_kwargs['attention_mask'] = torch.cat([attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    return model_kwargs
