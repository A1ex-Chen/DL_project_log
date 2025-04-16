def decode_latents(self, latents, cpu_vae=False):
    """Decodes a given array of latents into pixel space"""
    if cpu_vae:
        lat = deepcopy(latents).cpu()
        vae = deepcopy(self.vae).cpu()
    else:
        lat = latents
        vae = self.vae
    lat = 1 / 0.18215 * lat
    image = vae.decode(lat).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return self.numpy_to_pil(image)
