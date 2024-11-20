def _encode_image_prompt(self, prompt, device, num_images_per_prompt,
    do_classifier_free_guidance):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

    def normalize_embeddings(encoder_output):
        embeds = self.image_encoder.vision_model.post_layernorm(encoder_output
            .last_hidden_state)
        embeds = self.image_encoder.visual_projection(embeds)
        embeds_pooled = embeds[:, 0:1]
        embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
        return embeds
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    image_input = self.image_feature_extractor(images=prompt,
        return_tensors='pt')
    pixel_values = image_input.pixel_values.to(device).to(self.
        image_encoder.dtype)
    image_embeddings = self.image_encoder(pixel_values)
    image_embeddings = normalize_embeddings(image_embeddings)
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed *
        num_images_per_prompt, seq_len, -1)
    if do_classifier_free_guidance:
        uncond_images = [np.zeros((512, 512, 3)) + 0.5] * batch_size
        uncond_images = self.image_feature_extractor(images=uncond_images,
            return_tensors='pt')
        pixel_values = uncond_images.pixel_values.to(device).to(self.
            image_encoder.dtype)
        negative_prompt_embeds = self.image_encoder(pixel_values)
        negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_images_per_prompt, seq_len, -1)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings]
            )
    return image_embeddings
