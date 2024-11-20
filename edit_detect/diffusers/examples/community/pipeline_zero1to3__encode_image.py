def _encode_image(self, image, device, num_images_per_prompt,
    do_classifier_free_guidance):
    dtype = next(self.image_encoder.parameters()).dtype
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            assert image.shape[0
                ] == 3, 'Image outside a batch should be of shape (3, H, W)'
            image = image.unsqueeze(0)
        assert image.ndim == 4, 'Image must have 4 dimensions'
        if image.min() < -1 or image.max() > 1:
            raise ValueError('Image should be in [-1, 1] range')
    else:
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert('RGB'))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    image = image.to(device=device, dtype=dtype)
    image = self.CLIP_preprocess(image)
    image_embeddings = self.image_encoder(image).image_embeds.to(dtype=dtype)
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
