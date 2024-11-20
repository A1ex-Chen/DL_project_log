def _prepare_attention_mask_for_generation(self, input_ids: torch.Tensor,
    pad_token_id: int, eos_token_id: int) ->torch.LongTensor:
    is_pad_token_in_inputs_ids = (pad_token_id is not None and pad_token_id in
        input_ids)
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None or 
        eos_token_id is not None and pad_token_id != eos_token_id)
    if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
        return input_ids.ne(pad_token_id).long()
    return input_ids.new_ones(input_ids.shape)
