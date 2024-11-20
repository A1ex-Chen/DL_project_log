def prepare_image_latents(self, image, batch_size, dtype, device, generator
    =None):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
    image = image.to(device=device, dtype=dtype)
    if image.shape[1] == 4:
        latents = image
    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        if isinstance(generator, list):
            latents = [self.vae.encode(image[i:i + 1]).latent_dist.sample(
                generator[i]) for i in range(batch_size)]
            latents = torch.cat(latents, dim=0)
        else:
            latents = self.vae.encode(image).latent_dist.sample(generator)
        latents = self.vae.config.scaling_factor * latents
    if batch_size != latents.shape[0]:
        if batch_size % latents.shape[0] == 0:
            deprecation_message = (
                f'You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial images (`image`). Initial images are now duplicating to match the number of text prompts. Note that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update your script to pass as many initial images as text prompts to suppress this warning.'
                )
            deprecate('len(prompt) != len(image)', '1.0.0',
                deprecation_message, standard_warn=False)
            additional_latents_per_image = batch_size // latents.shape[0]
            latents = torch.cat([latents] * additional_latents_per_image, dim=0
                )
        else:
            raise ValueError(
                f'Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts.'
                )
    else:
        latents = torch.cat([latents], dim=0)
    return latents
