def _infer_batch_size(self, mode, prompt, prompt_embeds, image,
    num_images_per_prompt, num_prompts_per_image, latents, prompt_latents,
    vae_latents, clip_latents):
    """Infers the batch size and multiplier depending on mode and supplied arguments to `__call__`."""
    if num_images_per_prompt is None:
        num_images_per_prompt = 1
    if num_prompts_per_image is None:
        num_prompts_per_image = 1
    assert num_images_per_prompt > 0, 'num_images_per_prompt must be a positive integer'
    assert num_prompts_per_image > 0, 'num_prompts_per_image must be a positive integer'
    if mode in ['text2img']:
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        multiplier = num_images_per_prompt
    elif mode in ['img2text']:
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        else:
            batch_size = image.shape[0]
        multiplier = num_prompts_per_image
    elif mode in ['img']:
        if vae_latents is not None:
            batch_size = vae_latents.shape[0]
        elif clip_latents is not None:
            batch_size = clip_latents.shape[0]
        else:
            batch_size = 1
        multiplier = num_images_per_prompt
    elif mode in ['text']:
        if prompt_latents is not None:
            batch_size = prompt_latents.shape[0]
        else:
            batch_size = 1
        multiplier = num_prompts_per_image
    elif mode in ['joint']:
        if latents is not None:
            batch_size = latents.shape[0]
        elif prompt_latents is not None:
            batch_size = prompt_latents.shape[0]
        elif vae_latents is not None:
            batch_size = vae_latents.shape[0]
        elif clip_latents is not None:
            batch_size = clip_latents.shape[0]
        else:
            batch_size = 1
        if num_images_per_prompt == num_prompts_per_image:
            multiplier = num_images_per_prompt
        else:
            multiplier = min(num_images_per_prompt, num_prompts_per_image)
            logger.warning(
                f'You are using mode `{mode}` and `num_images_per_prompt`: {num_images_per_prompt} and num_prompts_per_image: {num_prompts_per_image} are not equal. Using batch size equal to `min(num_images_per_prompt, num_prompts_per_image) = {batch_size}.'
                )
    return batch_size, multiplier
