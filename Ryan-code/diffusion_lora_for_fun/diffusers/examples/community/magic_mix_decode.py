def decode(self, latent):
    latent = 1 / 0.18215 * latent
    with torch.no_grad():
        img = self.vae.decode(latent).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    img = (img * 255).round().astype('uint8')
    return Image.fromarray(img[0])
