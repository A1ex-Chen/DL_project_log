def encode(self, img):
    with torch.no_grad():
        latent = self.vae.encode(tfms.ToTensor()(img).unsqueeze(0).to(self.
            device) * 2 - 1)
        latent = 0.18215 * latent.latent_dist.sample()
    return latent
