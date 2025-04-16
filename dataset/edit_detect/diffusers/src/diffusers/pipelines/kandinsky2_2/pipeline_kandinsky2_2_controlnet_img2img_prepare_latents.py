def prepare_latents(self, image, timestep, batch_size,
    num_images_per_prompt, dtype, device, generator=None):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
    image = image.to(device=device, dtype=dtype)
    batch_size = batch_size * num_images_per_prompt
    if image.shape[1] == 4:
        init_latents = image
    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        elif isinstance(generator, list):
            init_latents = [self.movq.encode(image[i:i + 1]).latent_dist.
                sample(generator[i]) for i in range(batch_size)]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.movq.encode(image).latent_dist.sample(generator
                )
        init_latents = self.movq.config.scaling_factor * init_latents
    init_latents = torch.cat([init_latents], dim=0)
    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype
        )
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    return latents
