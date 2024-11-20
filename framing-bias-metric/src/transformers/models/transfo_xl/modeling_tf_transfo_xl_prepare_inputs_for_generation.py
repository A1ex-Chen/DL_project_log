def prepare_inputs_for_generation(self, inputs, past, **model_kwargs):
    inputs = {'input_ids': inputs}
    if past:
        inputs['mems'] = past
    return inputs
