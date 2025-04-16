def prepare_mask_latents(self, mask_image, batch_size, height, width, dtype,
    device, do_classifier_free_guidance):
    mask_image = F.interpolate(mask_image, size=(height // self.
        vae_scale_factor, width // self.vae_scale_factor))
    mask_image = mask_image.to(device=device, dtype=dtype)
    if mask_image.shape[0] < batch_size:
        if not batch_size % mask_image.shape[0] == 0:
            raise ValueError(
                f"The passed mask and the required batch size don't match. Masks are supposed to be duplicated to a total batch size of {batch_size}, but {mask_image.shape[0]} masks were passed. Make sure the number of masks that you pass is divisible by the total requested batch size."
                )
        mask_image = mask_image.repeat(batch_size // mask_image.shape[0], 1,
            1, 1)
    mask_image = torch.cat([mask_image] * 2
        ) if do_classifier_free_guidance else mask_image
    mask_image_latents = mask_image
    return mask_image_latents
