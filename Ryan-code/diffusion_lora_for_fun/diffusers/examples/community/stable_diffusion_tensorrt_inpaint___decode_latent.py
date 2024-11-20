def __decode_latent(self, latents):
    images = runEngine(self.engine['vae'], {'latent': device_view(latents)},
        self.stream)['images']
    images = (images / 2 + 0.5).clamp(0, 1)
    return images.cpu().permute(0, 2, 3, 1).float().numpy()
