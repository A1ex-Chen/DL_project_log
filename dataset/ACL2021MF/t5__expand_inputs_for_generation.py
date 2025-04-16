@staticmethod
def _expand_inputs_for_generation(input_ids: torch.LongTensor, expand_size:
    int=1, is_encoder_decoder: bool=False, attention_mask: torch.LongTensor
    =None, encoder_outputs: ModelOutput=None, **model_kwargs) ->Tuple[torch
    .LongTensor, Dict[str, Any]]:
    expanded_return_idx = torch.arange(input_ids.shape[0]).view(-1, 1).repeat(
        1, expand_size).view(-1).to(input_ids.device)
    input_ids = input_ids.index_select(0, expanded_return_idx)
    if attention_mask is not None:
        model_kwargs['attention_mask'] = attention_mask.index_select(0,
            expanded_return_idx)
    if is_encoder_decoder:
        assert encoder_outputs is not None
        encoder_outputs['last_hidden_state'
            ] = encoder_outputs.last_hidden_state.index_select(0,
            expanded_return_idx)
        model_kwargs['encoder_outputs'] = encoder_outputs
    for kwarg in ['decoder_copy_pos', 'decoder_concept_cls',
        'decoder_mention_flag', 'decoder_copy_mention_flag',
        'decoder_cls_on_input', 'encoder_img_mask']:
        if kwarg in model_kwargs:
            model_kwargs[kwarg] = model_kwargs[kwarg].index_select(0,
                expanded_return_idx)
    return input_ids, model_kwargs
