def prepare_mask_latents(self, mask, masked_image, batch_size, height,
    width, dtype, device, generator, do_classifier_free_guidance):
    mask = torch.nn.functional.interpolate(mask, size=(height // self.
        vae_scale_factor, width // self.vae_scale_factor))
    mask = mask.to(device=device, dtype=dtype)
    masked_image = masked_image.to(device=device, dtype=dtype)
    if masked_image.shape[1] == 4:
        masked_image_latents = masked_image
    else:
        masked_image_latents = self._encode_vae_image(masked_image,
            generator=generator)
    if mask.shape[0] < batch_size:
        if not batch_size % mask.shape[0] == 0:
            raise ValueError(
                f"The passed mask and the required batch size don't match. Masks are supposed to be duplicated to a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number of masks that you pass is divisible by the total requested batch size."
                )
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if masked_image_latents.shape[0] < batch_size:
        if not batch_size % masked_image_latents.shape[0] == 0:
            raise ValueError(
                f"The passed images and the required batch size don't match. Images are supposed to be duplicated to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed. Make sure the number of images that you pass is divisible by the total requested batch size."
                )
        masked_image_latents = masked_image_latents.repeat(batch_size //
            masked_image_latents.shape[0], 1, 1, 1)
    mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = torch.cat([masked_image_latents] * 2
        ) if do_classifier_free_guidance else masked_image_latents
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    return mask, masked_image_latents
