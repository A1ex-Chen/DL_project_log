def prepare_image_vae_latents(self, batch_size, num_prompts_per_image,
    num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
    shape = (batch_size * num_prompts_per_image, num_channels_latents, 
        height // self.vae_scale_factor, width // self.vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device,
            dtype=dtype)
    else:
        latents = latents.repeat(num_prompts_per_image, 1, 1, 1)
        latents = latents.to(device=device, dtype=dtype)
    latents = latents * self.scheduler.init_noise_sigma
    return latents
