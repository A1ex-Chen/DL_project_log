def prepare_latents(self, video, height, width, num_channels_latents,
    batch_size, timestep, dtype, device, generator, latents=None):
    if latents is None:
        num_frames = video.shape[1]
    else:
        num_frames = latents.shape[2]
    shape = (batch_size, num_channels_latents, num_frames, height // self.
        vae_scale_factor, width // self.vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    if latents is None:
        if self.vae.config.force_upcast:
            video = video.float()
            self.vae.to(dtype=torch.float32)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                    )
            init_latents = [retrieve_latents(self.vae.encode(video[i]),
                generator=generator[i]).unsqueeze(0) for i in range(batch_size)
                ]
        else:
            init_latents = [retrieve_latents(self.vae.encode(vid),
                generator=generator).unsqueeze(0) for vid in video]
        init_latents = torch.cat(init_latents, dim=0)
        if self.vae.config.force_upcast:
            self.vae.to(dtype)
        init_latents = init_latents.to(dtype)
        init_latents = self.vae.config.scaling_factor * init_latents
        if batch_size > init_latents.shape[0
            ] and batch_size % init_latents.shape[0] == 0:
            error_message = (
                f'You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial images (`image`). Please make sure to update your script to pass as many initial images as text prompts'
                )
            raise ValueError(error_message)
        elif batch_size > init_latents.shape[0
            ] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f'Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts.'
                )
        else:
            init_latents = torch.cat([init_latents], dim=0)
        noise = randn_tensor(init_latents.shape, generator=generator,
            device=device, dtype=dtype)
        latents = self.scheduler.add_noise(init_latents, noise, timestep
            ).permute(0, 2, 1, 3, 4)
    else:
        if shape != latents.shape:
            raise ValueError(
                f'`latents` expected to have shape={shape!r}, but found latents.shape={latents.shape!r}'
                )
        latents = latents.to(device, dtype=dtype)
    return latents
