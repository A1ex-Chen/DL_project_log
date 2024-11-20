def prepare_latents(self, image, timestep, batch_size, height, width, dtype,
    generator, latents=None):
    if image is None:
        shape = (batch_size, self.unet.config.in_channels, height // self.
            vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            latents = torch.randn(shape, generator=generator, device='cpu'
                ).numpy().astype(dtype)
        elif latents.shape != shape:
            raise ValueError(
                f'Unexpected latents shape, got {latents.shape}, expected {shape}'
                )
        latents = (torch.from_numpy(latents) * self.scheduler.init_noise_sigma
            ).numpy()
        return latents, None, None
    else:
        init_latents = self.vae_encoder(sample=image)[0]
        init_latents = 0.18215 * init_latents
        init_latents = np.concatenate([init_latents] * batch_size, axis=0)
        init_latents_orig = init_latents
        shape = init_latents.shape
        noise = torch.randn(shape, generator=generator, device='cpu').numpy(
            ).astype(dtype)
        latents = self.scheduler.add_noise(torch.from_numpy(init_latents),
            torch.from_numpy(noise), timestep).numpy()
        return latents, init_latents_orig, noise
