def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens
