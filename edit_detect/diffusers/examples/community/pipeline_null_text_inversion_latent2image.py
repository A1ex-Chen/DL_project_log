@torch.no_grad()
def latent2image(self, latents):
    latents = 1 / 0.18215 * latents.detach()
    image = self.vae.decode(latents)['sample'].detach()
    image = self.processor.postprocess(image, output_type='pil')[0]
    return image
