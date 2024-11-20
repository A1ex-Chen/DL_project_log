def decode_latents(self, latents):
    deprecation_message = (
        'The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead'
        )
    deprecate('decode_latents', '1.0.0', deprecation_message, standard_warn
        =False)
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image
