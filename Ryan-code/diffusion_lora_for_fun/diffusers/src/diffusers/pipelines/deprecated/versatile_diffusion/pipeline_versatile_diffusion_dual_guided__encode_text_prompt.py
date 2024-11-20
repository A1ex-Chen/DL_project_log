def _encode_text_prompt(self, prompt, device, num_images_per_prompt,
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
        embeds = self.text_encoder.text_projection(encoder_output.
            last_hidden_state)
        embeds_pooled = encoder_output.text_embeds
        embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1,
            keepdim=True)
        return embeds
    batch_size = len(prompt)
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
        uncond_tokens = [''] * batch_size
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
