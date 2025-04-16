def encode_image(self, image, device, num_images_per_prompt,
    output_hidden_states=None):
    dtype = next(self.image_encoder.parameters()).dtype
    if not isinstance(image, torch.Tensor):
        image = self.feature_extractor(image, return_tensors='pt').pixel_values
    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = self.image_encoder(image,
            output_hidden_states=True).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0)
        uncond_image_enc_hidden_states = self.image_encoder(torch.
            zeros_like(image), output_hidden_states=True).hidden_states[-2]
        uncond_image_enc_hidden_states = (uncond_image_enc_hidden_states.
            repeat_interleave(num_images_per_prompt, dim=0))
        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt,
            dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds
