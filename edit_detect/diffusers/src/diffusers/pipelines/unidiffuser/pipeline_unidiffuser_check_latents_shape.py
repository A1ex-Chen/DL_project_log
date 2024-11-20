def check_latents_shape(self, latents_name, latents, expected_shape):
    latents_shape = latents.shape
    expected_num_dims = len(expected_shape) + 1
    expected_shape_str = ', '.join(str(dim) for dim in expected_shape)
    if len(latents_shape) != expected_num_dims:
        raise ValueError(
            f'`{latents_name}` should have shape (batch_size, {expected_shape_str}), but the current shape {latents_shape} has {len(latents_shape)} dimensions.'
            )
    for i in range(1, expected_num_dims):
        if latents_shape[i] != expected_shape[i - 1]:
            raise ValueError(
                f'`{latents_name}` should have shape (batch_size, {expected_shape_str}), but the current shape {latents_shape} has {latents_shape[i]} != {expected_shape[i - 1]} at dimension {i}.'
                )
