def _shift_right(self, input_ids):
    decoder_start_token_id = self.config.decoder_start_token_id
    pad_token_id = self.config.pad_token_id
    assert decoder_start_token_id is not None, 'self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information'
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    assert torch.all(shifted_input_ids >= 0).item(
        ), 'Verify that `shifted_input_ids` has only positive values'
    return shifted_input_ids
