def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **
    model_kwargs):
    input_shape = input_ids.shape
    effective_batch_size = input_shape[0]
    assert self.config.pad_token_id is not None, 'The PAD token should be defined for generation'
    attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((
        attention_mask.shape[0], 1))], dim=-1)
    dummy_token = torch.full((effective_batch_size, 1), self.config.
        pad_token_id, dtype=torch.long, device=input_ids.device)
    input_ids = torch.cat([input_ids, dummy_token], dim=1)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}
