def shift_tokens_right(self, input_ids, start_token_id=None):
    """Shift input ids one token to the right, and pad with start_token_id"""
    if start_token_id is None:
        start_token_id = self.config.decoder_start_token_id
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = start_token_id
    return shifted_input_ids
