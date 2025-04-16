def _encode_image(self, image, device, batch_size, num_images_per_prompt,
    do_classifier_free_guidance, noise_level, generator, image_embeds):
    dtype = next(self.image_encoder.parameters()).dtype
    if isinstance(image, PIL.Image.Image):
        repeat_by = batch_size
    else:
        repeat_by = num_images_per_prompt
    if image_embeds is None:
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors='pt'
                ).pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
    image_embeds = self.noise_image_embeddings(image_embeds=image_embeds,
        noise_level=noise_level, generator=generator)
    image_embeds = image_embeds.unsqueeze(1)
    bs_embed, seq_len, _ = image_embeds.shape
    image_embeds = image_embeds.repeat(1, repeat_by, 1)
    image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
    image_embeds = image_embeds.squeeze(1)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeds)
        image_embeds = torch.cat([negative_prompt_embeds, image_embeds])
    return image_embeds
