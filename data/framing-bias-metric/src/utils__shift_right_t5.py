def _shift_right_t5(self, input_ids):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = self.pad_token_id
    return shifted_input_ids
