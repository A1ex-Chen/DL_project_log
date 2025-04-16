def prepare_inputs_for_generation(self, input_ids, past=None, **model_kwargs):
    inputs = {}
    if past:
        inputs['mems'] = past
        inputs['input_ids'] = input_ids[:, -1].unsqueeze(-1)
    else:
        inputs['input_ids'] = input_ids
    return inputs
