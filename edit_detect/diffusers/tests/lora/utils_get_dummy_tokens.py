def get_dummy_tokens(self):
    max_seq_length = 77
    inputs = torch.randint(2, 56, size=(1, max_seq_length), generator=torch
        .manual_seed(0))
    prepared_inputs = {}
    prepared_inputs['input_ids'] = inputs
    return prepared_inputs
