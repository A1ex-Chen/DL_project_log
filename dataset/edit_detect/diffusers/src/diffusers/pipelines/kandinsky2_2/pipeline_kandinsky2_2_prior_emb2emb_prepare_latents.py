def prepare_latents(self, emb, timestep, batch_size, num_images_per_prompt,
    dtype, device, generator=None):
    emb = emb.to(device=device, dtype=dtype)
    batch_size = batch_size * num_images_per_prompt
    init_latents = emb
    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0
        ] == 0:
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
