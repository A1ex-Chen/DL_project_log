def prepare_latents(self, batch_size, num_channels_latents, height, width,
    dtype, device, latents=None):
    shape = batch_size, num_channels_latents, int(height
        ) // self.vae_scale_factor, int(width) // self.vae_scale_factor
    if latents is None:
        latents = torch.randn(shape, dtype=dtype).to(device)
    else:
        latents = latents.to(device)
    latents = latents * self.scheduler.init_noise_sigma
    return latents
