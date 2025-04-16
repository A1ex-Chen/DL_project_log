def prepare_masked_condition(self, image, batch_size, num_channels_latents,
    num_frames, height, width, dtype, device, generator, motion_scale=0):
    shape = (batch_size, num_channels_latents, num_frames, height // self.
        vae_scale_factor, width // self.vae_scale_factor)
    _, _, _, scaled_height, scaled_width = shape
    image = self.video_processor.preprocess(image)
    image = image.to(device, dtype)
    if isinstance(generator, list):
        image_latent = [self.vae.encode(image[k:k + 1]).latent_dist.sample(
            generator[k]) for k in range(batch_size)]
        image_latent = torch.cat(image_latent, dim=0)
    else:
        image_latent = self.vae.encode(image).latent_dist.sample(generator)
    image_latent = image_latent.to(device=device, dtype=dtype)
    image_latent = torch.nn.functional.interpolate(image_latent, size=[
        scaled_height, scaled_width])
    image_latent_padding = image_latent.clone(
        ) * self.vae.config.scaling_factor
    mask = torch.zeros((batch_size, 1, num_frames, scaled_height, scaled_width)
        ).to(device=device, dtype=dtype)
    mask_coef = prepare_mask_coef_by_statistics(num_frames, 0, motion_scale)
    masked_image = torch.zeros(batch_size, 4, num_frames, scaled_height,
        scaled_width).to(device=device, dtype=self.unet.dtype)
    for f in range(num_frames):
        mask[:, :, f, :, :] = mask_coef[f]
        masked_image[:, :, f, :, :] = image_latent_padding.clone()
    mask = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask
    masked_image = torch.cat([masked_image] * 2
        ) if self.do_classifier_free_guidance else masked_image
    return mask, masked_image
