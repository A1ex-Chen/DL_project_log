def prepare_latents(self, image, timestep, num_images_per_prompt,
    batch_size, num_channels_latents, height, width, dtype, device,
    generator, latents=None):
    if image is None:
        batch_size = batch_size * num_images_per_prompt
        shape = batch_size, num_channels_latents, int(height
            ) // self.vae_scale_factor, int(width) // self.vae_scale_factor
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=
                device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents, None, None
    else:
        image = image.to(device=self.device, dtype=dtype)
        init_latent_dist = self.vae.encode(image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = self.vae.config.scaling_factor * init_latents
        init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)
        init_latents_orig = init_latents
        noise = randn_tensor(init_latents.shape, generator=generator,
            device=self.device, dtype=dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents
        return latents, init_latents_orig, noise
