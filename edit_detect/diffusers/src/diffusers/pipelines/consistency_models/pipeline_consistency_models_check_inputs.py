def check_inputs(self, num_inference_steps, timesteps, latents, batch_size,
    img_size, callback_steps):
    if num_inference_steps is None and timesteps is None:
        raise ValueError(
            'Exactly one of `num_inference_steps` or `timesteps` must be supplied.'
            )
    if num_inference_steps is not None and timesteps is not None:
        logger.warning(
            f'Both `num_inference_steps`: {num_inference_steps} and `timesteps`: {timesteps} are supplied; `timesteps` will be used over `num_inference_steps`.'
            )
    if latents is not None:
        expected_shape = batch_size, 3, img_size, img_size
        if latents.shape != expected_shape:
            raise ValueError(
                f'The shape of latents is {latents.shape} but is expected to be {expected_shape}.'
                )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
