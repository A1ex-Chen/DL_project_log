def prepare_inputs_for_generation(self, input_ids, past=None, use_mems=None,
    **kwargs):
    effective_batch_size = input_ids.shape[0]
    dummy_token = torch.zeros((effective_batch_size, 1), dtype=torch.long,
        device=input_ids.device)
    offset = 2
    if past:
        input_ids = torch.cat([input_ids[:, -offset:], dummy_token], dim=1)
    else:
        input_ids = torch.cat([input_ids, dummy_token], dim=1)
    sequence_length = input_ids.shape[1]
    perm_mask = torch.zeros((effective_batch_size, sequence_length,
        sequence_length), dtype=torch.float, device=input_ids.device)
    perm_mask[:, :, -1] = 1.0
    target_mapping = torch.zeros((effective_batch_size, 1, sequence_length),
        dtype=torch.float, device=input_ids.device)
    target_mapping[:, 0, -1] = 1.0
    inputs = {'input_ids': input_ids, 'perm_mask': perm_mask,
        'target_mapping': target_mapping, 'use_mems': use_mems}
    if past:
        inputs['mems'] = tuple(layer_past[:-offset, :, :] for layer_past in
            past)
    return inputs
