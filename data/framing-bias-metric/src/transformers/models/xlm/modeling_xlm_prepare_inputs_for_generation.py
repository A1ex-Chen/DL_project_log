def prepare_inputs_for_generation(self, input_ids, **kwargs):
    mask_token_id = self.config.mask_token_id
    lang_id = self.config.lang_id
    effective_batch_size = input_ids.shape[0]
    mask_token = torch.full((effective_batch_size, 1), mask_token_id, dtype
        =torch.long, device=input_ids.device)
    input_ids = torch.cat([input_ids, mask_token], dim=1)
    if lang_id is not None:
        langs = torch.full_like(input_ids, lang_id)
    else:
        langs = None
    return {'input_ids': input_ids, 'langs': langs}
