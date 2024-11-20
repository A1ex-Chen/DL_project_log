def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
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
    return latents
