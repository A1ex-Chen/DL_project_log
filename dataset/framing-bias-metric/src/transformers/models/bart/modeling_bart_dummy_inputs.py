@property
def dummy_inputs(self):
    pad_token = self.config.pad_token_id
    input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]],
        device=self.device)
    dummy_inputs = {'attention_mask': input_ids.ne(pad_token), 'input_ids':
        input_ids}
    return dummy_inputs
