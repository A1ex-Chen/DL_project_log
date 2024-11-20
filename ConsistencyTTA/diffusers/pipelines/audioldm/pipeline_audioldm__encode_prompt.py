def _encode_prompt(self, prompt, device, num_waveforms_per_prompt,
    do_classifier_free_guidance, negative_prompt=None, prompt_embeds:
    Optional[torch.FloatTensor]=None, negative_prompt_embeds: Optional[
    torch.FloatTensor]=None):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    if prompt_embeds is None:
        text_inputs = self.tokenizer(prompt, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding='longest',
            return_tensors='pt').input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
            ] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, 
                self.tokenizer.model_max_length - 1:-1])
            logger.warning(
                f'The following part of your input was truncated because CLAP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
                )
        prompt_embeds = self.text_encoder(text_input_ids.to(device),
            attention_mask=attention_mask.to(device))
        prompt_embeds = prompt_embeds.text_embeds
        prompt_embeds = F.normalize(prompt_embeds, dim=-1)
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=
        device)
    bs_embed, seq_len = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
    prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt,
        seq_len)
    if do_classifier_free_guidance and negative_prompt_embeds is None:
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
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='pt')
        uncond_input_ids = uncond_input.input_ids.to(device)
        attention_mask = uncond_input.attention_mask.to(device)
        negative_prompt_embeds = self.text_encoder(uncond_input_ids,
            attention_mask=attention_mask)
        negative_prompt_embeds = negative_prompt_embeds.text_embeds
        negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)
    if do_classifier_free_guidance:
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.
            text_encoder.dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_waveforms_per_prompt)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_waveforms_per_prompt, seq_len)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds