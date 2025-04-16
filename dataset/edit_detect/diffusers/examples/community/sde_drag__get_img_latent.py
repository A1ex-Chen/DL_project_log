@torch.no_grad()
def _get_img_latent(self, image, height=None, weight=None):
    data = image.convert('RGB')
    if height is not None:
        data = data.resize((weight, height))
    transform = transforms.ToTensor()
    data = transform(data).unsqueeze(0)
    data = data * 2.0 - 1.0
    data = data.to(self.device, dtype=self.vae.dtype)
    latent = self.vae.encode(data).latent_dist.sample()
    latent = 0.18215 * latent
    return latent
