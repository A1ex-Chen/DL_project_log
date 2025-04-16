def callback_inputs_test(pipe, i, t, callback_kwargs):
    missing_callback_inputs = set()
    for v in pipe._callback_tensor_inputs:
        if v not in callback_kwargs:
            missing_callback_inputs.add(v)
    self.assertTrue(len(missing_callback_inputs) == 0,
        f'Missing callback tensor inputs: {missing_callback_inputs}')
    last_i = pipe.num_timesteps - 1
    if i == last_i:
        callback_kwargs['denoised'] = torch.zeros_like(callback_kwargs[
            'denoised'])
    return callback_kwargs
