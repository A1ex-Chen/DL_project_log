@torch.no_grad()
def image2latent(self, image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
    latents = self.vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents
