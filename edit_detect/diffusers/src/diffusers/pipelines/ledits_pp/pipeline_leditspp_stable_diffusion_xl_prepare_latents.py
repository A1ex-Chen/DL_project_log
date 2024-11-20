def prepare_latents(self, device, latents):
    latents = latents.to(device)
    latents = latents * self.scheduler.init_noise_sigma
    return latents
