def _encode_prompt(self, prompt, num_images_per_prompt,
    do_classifier_free_guidance, negative_prompt):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='np')
    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer(prompt, padding='max_length',
        return_tensors='np').input_ids
    if not np.array_equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.
            tokenizer.model_max_length - 1:-1])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
            )
    prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32)
        )[0]
    prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt] * batch_size
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
        else:
            uncond_tokens = negative_prompt
        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='np')
        negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.
            input_ids.astype(np.int32))[0]
        negative_prompt_embeds = np.repeat(negative_prompt_embeds,
            num_images_per_prompt, axis=0)
        prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds
