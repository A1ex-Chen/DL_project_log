def prepare_image_latents(self, image, batch_size, num_images_per_prompt,
    dtype, device, do_classifier_free_guidance, generator=None):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
    image = image.to(device=device, dtype=dtype)
    batch_size = batch_size * num_images_per_prompt
    if image.shape[1] == 4:
        image_latents = image
    else:
        needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.
            config.force_upcast)
        if needs_upcasting:
            image = image.float()
            self.upcast_vae()
        image_latents = retrieve_latents(self.vae.encode(image),
            sample_mode='argmax')
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    if batch_size > image_latents.shape[0
        ] and batch_size % image_latents.shape[0] == 0:
        deprecation_message = (
            f'You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial images (`image`). Initial images are now duplicating to match the number of text prompts. Note that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update your script to pass as many initial images as text prompts to suppress this warning.'
            )
        deprecate('len(prompt) != len(image)', '1.0.0', deprecation_message,
            standard_warn=False)
        additional_image_per_prompt = batch_size // image_latents.shape[0]
        image_latents = torch.cat([image_latents] *
            additional_image_per_prompt, dim=0)
    elif batch_size > image_latents.shape[0
        ] and batch_size % image_latents.shape[0] != 0:
        raise ValueError(
            f'Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts.'
            )
    else:
        image_latents = torch.cat([image_latents], dim=0)
    if do_classifier_free_guidance:
        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents,
            uncond_image_latents], dim=0)
    if image_latents.dtype != self.vae.dtype:
        image_latents = image_latents.to(dtype=self.vae.dtype)
    if self.is_cosxl_edit:
        image_latents = image_latents * self.vae.config.scaling_factor
    return image_latents
