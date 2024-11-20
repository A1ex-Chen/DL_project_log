def prepare_latents(self, latents, latent_timestep, shape, dtype, device,
    generator, scheduler):
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device,
            dtype=dtype)
    else:
        if latents.shape != shape:
            raise ValueError(
                f'Unexpected latents shape, got {latents.shape}, expected {shape}'
                )
        latents = latents.to(device)
    latents = latents * scheduler.init_noise_sigma
    shape = latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype
        )
    latents = self.add_noise(latents, noise, latent_timestep)
    return latents
