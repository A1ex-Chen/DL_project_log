def prepare_ref_latents(self, refimage, batch_size, dtype, device,
    generator, do_classifier_free_guidance):
    refimage = refimage.to(device=device, dtype=dtype)
    if isinstance(generator, list):
        ref_image_latents = [self.vae.encode(refimage[i:i + 1]).latent_dist
            .sample(generator=generator[i]) for i in range(batch_size)]
        ref_image_latents = torch.cat(ref_image_latents, dim=0)
    else:
        ref_image_latents = self.vae.encode(refimage).latent_dist.sample(
            generator=generator)
    ref_image_latents = self.vae.config.scaling_factor * ref_image_latents
    if ref_image_latents.shape[0] < batch_size:
        if not batch_size % ref_image_latents.shape[0] == 0:
            raise ValueError(
                f"The passed images and the required batch size don't match. Images are supposed to be duplicated to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed. Make sure the number of images that you pass is divisible by the total requested batch size."
                )
        ref_image_latents = ref_image_latents.repeat(batch_size //
            ref_image_latents.shape[0], 1, 1, 1)
    ref_image_latents = torch.cat([ref_image_latents] * 2
        ) if do_classifier_free_guidance else ref_image_latents
    ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
    return ref_image_latents
