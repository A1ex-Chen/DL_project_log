def prepare_latents(self, video, timestep, batch_size, dtype, device,
    generator=None):
    video = video.to(device=device, dtype=dtype)
    bsz, channel, frames, width, height = video.shape
    video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel,
        width, height)
    if video.shape[1] == 4:
        init_latents = video
    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        elif isinstance(generator, list):
            init_latents = [retrieve_latents(self.vae.encode(video[i:i + 1]
                ), generator=generator[i]) for i in range(batch_size)]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(video),
                generator=generator)
        init_latents = self.vae.config.scaling_factor * init_latents
    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0
        ] != 0:
        raise ValueError(
            f'Cannot duplicate `video` of batch size {init_latents.shape[0]} to {batch_size} text prompts.'
            )
    else:
        init_latents = torch.cat([init_latents], dim=0)
    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype
        )
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    latents = latents[None, :].reshape((bsz, frames, latents.shape[1]) +
        latents.shape[2:]).permute(0, 2, 1, 3, 4)
    return latents
