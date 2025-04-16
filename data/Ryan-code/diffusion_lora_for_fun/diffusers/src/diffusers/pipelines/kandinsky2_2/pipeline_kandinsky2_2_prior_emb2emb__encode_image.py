def _encode_image(self, image: Union[torch.Tensor, List[PIL.Image.Image]],
    device, num_images_per_prompt):
    if not isinstance(image, torch.Tensor):
        image = self.image_processor(image, return_tensors='pt'
            ).pixel_values.to(dtype=self.image_encoder.dtype, device=device)
    image_emb = self.image_encoder(image)['image_embeds']
    image_emb = image_emb.repeat_interleave(num_images_per_prompt, dim=0)
    image_emb.to(device=device)
    return image_emb
