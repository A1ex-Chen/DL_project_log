def decode_latents(self, latents):
    latents = 1 / 0.18215 * latents
    image = np.concatenate([self.vae_decoder(latent_sample=latents[i:i + 1]
        )[0] for i in range(latents.shape[0])])
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))
    return image
