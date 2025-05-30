def decode_latents(self, latents):
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image
