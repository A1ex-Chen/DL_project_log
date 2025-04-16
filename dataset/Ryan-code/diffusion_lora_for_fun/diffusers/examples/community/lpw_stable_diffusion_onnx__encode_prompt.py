def _encode_prompt(self, prompt, num_images_per_prompt,
    do_classifier_free_guidance, negative_prompt, max_embeddings_multiples):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    if negative_prompt is None:
        negative_prompt = [''] * batch_size
    elif isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
    if batch_size != len(negative_prompt):
        raise ValueError(
            f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
            )
    text_embeddings, uncond_embeddings = get_weighted_text_embeddings(pipe=
        self, prompt=prompt, uncond_prompt=negative_prompt if
        do_classifier_free_guidance else None, max_embeddings_multiples=
        max_embeddings_multiples)
    text_embeddings = text_embeddings.repeat(num_images_per_prompt, 0)
    if do_classifier_free_guidance:
        uncond_embeddings = uncond_embeddings.repeat(num_images_per_prompt, 0)
        text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
    return text_embeddings
