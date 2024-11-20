@torch.no_grad()
def decode_latents(self, latents: torch.Tensor):
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image
