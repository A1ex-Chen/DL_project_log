def decode_latents(self, latents):
    latents = 1 / 0.08333 * latents
    image = self.vae(latent_sample=latents)[0]
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))
    return image
