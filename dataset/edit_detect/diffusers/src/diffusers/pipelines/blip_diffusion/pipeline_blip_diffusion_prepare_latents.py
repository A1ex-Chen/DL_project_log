def prepare_latents(self, batch_size, num_channels, height, width, dtype,
    device, generator, latents=None):
    shape = batch_size, num_channels, height, width
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device,
            dtype=dtype)
    else:
        latents = latents.to(device=device, dtype=dtype)
    latents = latents * self.scheduler.init_noise_sigma
    return latents
