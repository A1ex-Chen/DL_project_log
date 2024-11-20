def _encode_image(self, image, device, num_images_per_prompt,
    image_embeddings: Optional[torch.Tensor]=None):
    dtype = next(self.image_encoder.parameters()).dtype
    if image_embeddings is None:
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors='pt'
                ).pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
    image_embeddings = image_embeddings.repeat_interleave(num_images_per_prompt
        , dim=0)
    return image_embeddings
