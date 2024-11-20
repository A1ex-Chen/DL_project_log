def encode_image(self, images, device, dtype, batch_size, num_images_per_prompt
    ):
    image_embeds = []
    for image in images:
        image = self.feature_extractor(image, return_tensors='pt').pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embed = self.image_encoder(image).image_embeds.unsqueeze(1)
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds, dim=1)
    image_embeds = image_embeds.repeat(batch_size * num_images_per_prompt, 1, 1
        )
    negative_image_embeds = torch.zeros_like(image_embeds)
    return image_embeds, negative_image_embeds
