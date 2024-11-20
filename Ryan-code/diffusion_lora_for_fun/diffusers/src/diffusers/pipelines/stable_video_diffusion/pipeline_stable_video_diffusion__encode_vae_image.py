def _encode_vae_image(self, image: torch.Tensor, device: Union[str, torch.
    device], num_videos_per_prompt: int, do_classifier_free_guidance: bool):
    image = image.to(device=device)
    image_latents = self.vae.encode(image).latent_dist.mode()
    image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
    if do_classifier_free_guidance:
        negative_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([negative_image_latents, image_latents])
    return image_latents
