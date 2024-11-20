@property
def dummy_inputs(self):
    input_ids = torch.tensor(DUMMY_INPUTS)
    input_mask = torch.tensor(DUMMY_MASK)
    dummy_inputs = {'decoder_input_ids': input_ids, 'input_ids': input_ids,
        'decoder_attention_mask': input_mask}
    return dummy_inputs
