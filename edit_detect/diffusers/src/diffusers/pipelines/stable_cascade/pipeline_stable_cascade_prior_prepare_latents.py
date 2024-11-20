def prepare_latents(self, batch_size, height, width, num_images_per_prompt,
    dtype, device, generator, latents, scheduler):
    latent_shape = (num_images_per_prompt * batch_size, self.prior.config.
        in_channels, ceil(height / self.config.resolution_multiple), ceil(
        width / self.config.resolution_multiple))
    if latents is None:
        latents = randn_tensor(latent_shape, generator=generator, device=
            device, dtype=dtype)
    else:
        if latents.shape != latent_shape:
            raise ValueError(
                f'Unexpected latents shape, got {latents.shape}, expected {latent_shape}'
                )
        latents = latents.to(device)
    latents = latents * scheduler.init_noise_sigma
    return latents
