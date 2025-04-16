def prepare_latents(self, batch_size, num_channels_latents, height, width,
    dtype, device, latents):
    latents = latents.to(device)
    latents = latents * self.scheduler.init_noise_sigma
    return latents
