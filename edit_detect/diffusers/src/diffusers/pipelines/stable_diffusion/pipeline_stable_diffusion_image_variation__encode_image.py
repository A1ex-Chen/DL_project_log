def _encode_image(self, image, device, num_images_per_prompt,
    do_classifier_free_guidance):
    dtype = next(self.image_encoder.parameters()).dtype
    if not isinstance(image, torch.Tensor):
        image = self.feature_extractor(images=image, return_tensors='pt'
            ).pixel_values
    image = image.to(device=device, dtype=dtype)
    image_embeddings = self.image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed *
        num_images_per_prompt, seq_len, -1)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings]
            )
    return image_embeddings
