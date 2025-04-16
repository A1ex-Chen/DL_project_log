def _encode_image(self, image, device, num_images_per_prompt,
    do_classifier_free_guidance):
    if isinstance(image, List) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, axis=0) if image[0
            ].ndim == 4 else torch.stack(image, axis=0)
    if not isinstance(image, torch.Tensor):
        image = self.image_processor(image, return_tensors='pt').pixel_values[0
            ].unsqueeze(0)
    image = image.to(dtype=self.image_encoder.dtype, device=device)
    image_embeds = self.image_encoder(image)['last_hidden_state']
    image_embeds = image_embeds[:, 1:, :].contiguous()
    image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    if do_classifier_free_guidance:
        negative_image_embeds = torch.zeros_like(image_embeds)
        image_embeds = torch.cat([negative_image_embeds, image_embeds])
    return image_embeds
