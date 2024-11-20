def prepare_latents(self, image, timestep, batch_size,
    num_images_per_prompt, dtype, device, generator):
    image = image.to(device=self.device, dtype=dtype)
    init_latent_dist = self.vae.encode(image).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = self.vae.config.scaling_factor * init_latents
    init_latents = torch.cat([init_latents] * batch_size *
        num_images_per_prompt, dim=0)
    init_latents_orig = init_latents
    noise = randn_tensor(init_latents.shape, generator=generator, device=
        self.device, dtype=dtype)
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    return latents, init_latents_orig, noise
