def _infer_mode(self, prompt, prompt_embeds, image, latents, prompt_latents,
    vae_latents, clip_latents):
    """
        Infer the generation task ('mode') from the inputs to `__call__`. If the mode has been manually set, the set
        mode will be used.
        """
    prompt_available = prompt is not None or prompt_embeds is not None
    image_available = image is not None
    input_available = prompt_available or image_available
    prompt_latents_available = prompt_latents is not None
    vae_latents_available = vae_latents is not None
    clip_latents_available = clip_latents is not None
    full_latents_available = latents is not None
    image_latents_available = vae_latents_available and clip_latents_available
    all_indv_latents_available = (prompt_latents_available and
        image_latents_available)
    if self.mode is not None:
        mode = self.mode
    elif prompt_available:
        mode = 'text2img'
    elif image_available:
        mode = 'img2text'
    elif full_latents_available or all_indv_latents_available:
        mode = 'joint'
    elif prompt_latents_available:
        mode = 'text'
    elif image_latents_available:
        mode = 'img'
    else:
        mode = 'joint'
    if self.mode is None and prompt_available and image_available:
        logger.warning(
            f"You have supplied both a text prompt and image to the pipeline and mode has not been set manually, defaulting to mode '{mode}'."
            )
    if self.mode is None and not input_available:
        if vae_latents_available != clip_latents_available:
            logger.warning(
                f"You have supplied exactly one of `vae_latents` and `clip_latents`, whereas either both or none are expected to be supplied. Defaulting to mode '{mode}'."
                )
        elif not prompt_latents_available and not vae_latents_available and not clip_latents_available:
            logger.warning(
                f"No inputs or latents have been supplied, and mode has not been manually set, defaulting to mode '{mode}'."
                )
    return mode
