def prepare_latents(self, image, timestep, batch_size, num_channels_latents,
    height, width, dtype, device, latents=None, generator=None):
    shape = batch_size, num_channels_latents, int(height
        ) // self.vae_scale_factor, int(width) // self.vae_scale_factor
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
    image = image.to(device=device, dtype=dtype)
    if image.shape[1] == 4:
        init_latents = image
    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        elif isinstance(generator, list):
            init_latents = [self.vae.encode(image[i:i + 1]).latent_dist.
                sample(generator[i]) for i in range(batch_size)]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)
        init_latents = self.vae.config.scaling_factor * init_latents
    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0
        ] == 0:
        f"""You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial images (`image`). Initial images are now duplicating to match the number of text prompts. Note that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update your script to pass as many initial images as text prompts to suppress this warning."""
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] *
            additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0
        ] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f'Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts.'
            )
    else:
        init_latents = torch.cat([init_latents], dim=0)
    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype
        )
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    return latents
