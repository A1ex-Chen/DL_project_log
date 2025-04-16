def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
    is_last = i == pipe.num_timesteps - 1
    if is_last:
        callback_kwargs['latents'] = torch.zeros_like(callback_kwargs[
            'latents'])
    return callback_kwargs
