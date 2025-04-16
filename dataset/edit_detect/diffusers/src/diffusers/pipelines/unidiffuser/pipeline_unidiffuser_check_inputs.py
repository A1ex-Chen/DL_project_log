def check_inputs(self, mode, prompt, image, height, width, callback_steps,
    negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
    latents=None, prompt_latents=None, vae_latents=None, clip_latents=None):
    if (height % self.vae_scale_factor != 0 or width % self.
        vae_scale_factor != 0):
        raise ValueError(
            f'`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}.'
            )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    if mode == 'text2img':
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
                )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
                )
        elif prompt is not None and (not isinstance(prompt, str) and not
            isinstance(prompt, list)):
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
                )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
                )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                    )
    if mode == 'img2text':
        if image is None:
            raise ValueError(
                '`img2text` mode requires an image to be provided.')
    latent_height = height // self.vae_scale_factor
    latent_width = width // self.vae_scale_factor
    full_latents_available = latents is not None
    prompt_latents_available = prompt_latents is not None
    vae_latents_available = vae_latents is not None
    clip_latents_available = clip_latents is not None
    if full_latents_available:
        individual_latents_available = (prompt_latents is not None or 
            vae_latents is not None or clip_latents is not None)
        if individual_latents_available:
            logger.warning(
                'You have supplied both `latents` and at least one of `prompt_latents`, `vae_latents`, and `clip_latents`. The value of `latents` will override the value of any individually supplied latents.'
                )
        img_vae_dim = self.num_channels_latents * latent_height * latent_width
        text_dim = self.text_encoder_seq_len * self.text_encoder_hidden_size
        latents_dim = (img_vae_dim + self.image_encoder_projection_dim +
            text_dim)
        latents_expected_shape = latents_dim,
        self.check_latents_shape('latents', latents, latents_expected_shape)
    if prompt_latents_available:
        prompt_latents_expected_shape = (self.text_encoder_seq_len, self.
            text_encoder_hidden_size)
        self.check_latents_shape('prompt_latents', prompt_latents,
            prompt_latents_expected_shape)
    if vae_latents_available:
        vae_latents_expected_shape = (self.num_channels_latents,
            latent_height, latent_width)
        self.check_latents_shape('vae_latents', vae_latents,
            vae_latents_expected_shape)
    if clip_latents_available:
        clip_latents_expected_shape = 1, self.image_encoder_projection_dim
        self.check_latents_shape('clip_latents', clip_latents,
            clip_latents_expected_shape)
    if mode in ['text2img', 'img'
        ] and vae_latents_available and clip_latents_available:
        if vae_latents.shape[0] != clip_latents.shape[0]:
            raise ValueError(
                f'Both `vae_latents` and `clip_latents` are supplied, but their batch dimensions are not equal: {vae_latents.shape[0]} != {clip_latents.shape[0]}.'
                )
    if (mode == 'joint' and prompt_latents_available and
        vae_latents_available and clip_latents_available):
        if prompt_latents.shape[0] != vae_latents.shape[0
            ] or prompt_latents.shape[0] != clip_latents.shape[0]:
            raise ValueError(
                f'All of `prompt_latents`, `vae_latents`, and `clip_latents` are supplied, but their batch dimensions are not equal: {prompt_latents.shape[0]} != {vae_latents.shape[0]} != {clip_latents.shape[0]}.'
                )
