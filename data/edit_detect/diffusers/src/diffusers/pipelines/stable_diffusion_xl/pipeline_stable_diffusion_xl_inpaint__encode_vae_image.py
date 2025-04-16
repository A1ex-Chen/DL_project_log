def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    dtype = image.dtype
    if self.vae.config.force_upcast:
        image = image.float()
        self.vae.to(dtype=torch.float32)
    if isinstance(generator, list):
        image_latents = [retrieve_latents(self.vae.encode(image[i:i + 1]),
            generator=generator[i]) for i in range(image.shape[0])]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(self.vae.encode(image), generator=
            generator)
    if self.vae.config.force_upcast:
        self.vae.to(dtype)
    image_latents = image_latents.to(dtype)
    image_latents = self.vae.config.scaling_factor * image_latents
    return image_latents
