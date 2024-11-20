def prepare_intermediate_images(self, batch_size, num_channels, height,
    width, dtype, device, generator):
    shape = batch_size, num_channels, height, width
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    intermediate_images = randn_tensor(shape, generator=generator, device=
        device, dtype=dtype)
    intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
    return intermediate_images
