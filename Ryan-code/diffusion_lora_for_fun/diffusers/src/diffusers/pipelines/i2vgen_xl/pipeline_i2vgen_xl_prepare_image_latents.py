def prepare_image_latents(self, image, device, num_frames,
    num_videos_per_prompt):
    image = image.to(device=device)
    image_latents = self.vae.encode(image).latent_dist.sample()
    image_latents = image_latents * self.vae.config.scaling_factor
    image_latents = image_latents.unsqueeze(2)
    frame_position_mask = []
    for frame_idx in range(num_frames - 1):
        scale = (frame_idx + 1) / (num_frames - 1)
        frame_position_mask.append(torch.ones_like(image_latents[:, :, :1]) *
            scale)
    if frame_position_mask:
        frame_position_mask = torch.cat(frame_position_mask, dim=2)
        image_latents = torch.cat([image_latents, frame_position_mask], dim=2)
    image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)
    if self.do_classifier_free_guidance:
        image_latents = torch.cat([image_latents] * 2)
    return image_latents
