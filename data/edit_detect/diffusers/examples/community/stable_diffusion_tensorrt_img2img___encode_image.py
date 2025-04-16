def __encode_image(self, init_image):
    init_latents = runEngine(self.engine['vae_encoder'], {'images':
        device_view(init_image)}, self.stream)['latent']
    init_latents = 0.18215 * init_latents
    return init_latents
