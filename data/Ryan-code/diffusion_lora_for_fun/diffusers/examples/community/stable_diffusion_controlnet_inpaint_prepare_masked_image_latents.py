def prepare_masked_image_latents(self, masked_image, batch_size, height,
    width, dtype, device, generator, do_classifier_free_guidance):
    masked_image = masked_image.to(device=device, dtype=dtype)
    if isinstance(generator, list):
        masked_image_latents = [self.vae.encode(masked_image[i:i + 1]).
            latent_dist.sample(generator=generator[i]) for i in range(
            batch_size)]
        masked_image_latents = torch.cat(masked_image_latents, dim=0)
    else:
        masked_image_latents = self.vae.encode(masked_image
            ).latent_dist.sample(generator=generator)
    masked_image_latents = (self.vae.config.scaling_factor *
        masked_image_latents)
    if masked_image_latents.shape[0] < batch_size:
        if not batch_size % masked_image_latents.shape[0] == 0:
            raise ValueError(
                f"The passed images and the required batch size don't match. Images are supposed to be duplicated to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed. Make sure the number of images that you pass is divisible by the total requested batch size."
                )
        masked_image_latents = masked_image_latents.repeat(batch_size //
            masked_image_latents.shape[0], 1, 1, 1)
    masked_image_latents = torch.cat([masked_image_latents] * 2
        ) if do_classifier_free_guidance else masked_image_latents
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    return masked_image_latents
