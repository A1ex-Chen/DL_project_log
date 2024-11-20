def _encode_prompt(self, prompt, device, num_images_per_prompt,
    do_classifier_free_guidance, negative_prompt):
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
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """

    def normalize_embeddings(encoder_output):
        embeds = self.text_encoder.text_projection(encoder_output.
            last_hidden_state)
        embeds_pooled = encoder_output.text_embeds
        embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1,
            keepdim=True)
        return embeds
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer(prompt, padding='max_length',
        return_tensors='pt').input_ids
    if not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.
            tokenizer.model_max_length - 1:-1])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
            )
    if hasattr(self.text_encoder.config, 'use_attention_mask'
        ) and self.text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None
    prompt_embeds = self.text_encoder(text_input_ids.to(device),
        attention_mask=attention_mask)
    prompt_embeds = normalize_embeddings(prompt_embeds)
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt,
        seq_len, -1)
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
        else:
            uncond_tokens = negative_prompt
        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='pt')
        if hasattr(self.text_encoder.config, 'use_attention_mask'
            ) and self.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None
        negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.
            to(device), attention_mask=attention_mask)
        negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_images_per_prompt, seq_len, -1)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds
