def prepare_latents(self, image, timestep, batch_size, dtype, device,
    generator=None):
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor` but is {type(image)}')
    image = image.to(device=device, dtype=dtype)
    if isinstance(generator, list):
        init_latents = [self.vae.encode(image[i:i + 1]).latent_dist.sample(
            generator[i]) for i in range(batch_size)]
        init_latents = torch.cat(init_latents, dim=0)
    else:
        init_latents = self.vae.encode(image).latent_dist.sample(generator)
    init_latents = 0.18215 * init_latents
    init_latents = init_latents.repeat_interleave(batch_size, dim=0)
    noise = randn_tensor(init_latents.shape, generator=generator, device=
        device, dtype=dtype)
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    return latents
