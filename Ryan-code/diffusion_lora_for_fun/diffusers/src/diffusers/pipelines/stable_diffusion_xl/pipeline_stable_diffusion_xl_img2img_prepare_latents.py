def prepare_latents(self, image, timestep, batch_size,
    num_images_per_prompt, dtype, device, generator=None, add_noise=True):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
    latents_mean = latents_std = None
    if hasattr(self.vae.config, 'latents_mean'
        ) and self.vae.config.latents_mean is not None:
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4,
            1, 1)
    if hasattr(self.vae.config, 'latents_std'
        ) and self.vae.config.latents_std is not None:
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1
            )
    if hasattr(self, 'final_offload_hook'
        ) and self.final_offload_hook is not None:
        self.text_encoder_2.to('cpu')
        torch.cuda.empty_cache()
    image = image.to(device=device, dtype=dtype)
    batch_size = batch_size * num_images_per_prompt
    if image.shape[1] == 4:
        init_latents = image
    else:
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        elif isinstance(generator, list):
            init_latents = [retrieve_latents(self.vae.encode(image[i:i + 1]
                ), generator=generator[i]) for i in range(batch_size)]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(image),
                generator=generator)
        if self.vae.config.force_upcast:
            self.vae.to(dtype)
        init_latents = init_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=self.device, dtype=dtype)
            latents_std = latents_std.to(device=self.device, dtype=dtype)
            init_latents = (init_latents - latents_mean
                ) * self.vae.config.scaling_factor / latents_std
        else:
            init_latents = self.vae.config.scaling_factor * init_latents
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
    if add_noise:
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device,
            dtype=dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    return latents
