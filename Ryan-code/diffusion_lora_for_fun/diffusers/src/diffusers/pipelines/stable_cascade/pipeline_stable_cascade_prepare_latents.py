def prepare_latents(self, batch_size, image_embeddings,
    num_images_per_prompt, dtype, device, generator, latents, scheduler):
    _, channels, height, width = image_embeddings.shape
    latents_shape = batch_size * num_images_per_prompt, 4, int(height *
        self.config.latent_dim_scale), int(width * self.config.latent_dim_scale
        )
    if latents is None:
        latents = randn_tensor(latents_shape, generator=generator, device=
            device, dtype=dtype)
    else:
        if latents.shape != latents_shape:
            raise ValueError(
                f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}'
                )
        latents = latents.to(device)
    latents = latents * scheduler.init_noise_sigma
    return latents
