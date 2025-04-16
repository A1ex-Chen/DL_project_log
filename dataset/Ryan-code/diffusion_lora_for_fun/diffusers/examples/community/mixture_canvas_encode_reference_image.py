def encode_reference_image(self, encoder, device, generator, cpu_vae=False):
    """Encodes the reference image for this Image2Image region into the latent space"""
    if cpu_vae:
        self.reference_latents = encoder.cpu().encode(self.reference_image
            ).latent_dist.mean.to(device)
    else:
        self.reference_latents = encoder.encode(self.reference_image.to(device)
            ).latent_dist.sample(generator=generator)
    self.reference_latents = 0.18215 * self.reference_latents
