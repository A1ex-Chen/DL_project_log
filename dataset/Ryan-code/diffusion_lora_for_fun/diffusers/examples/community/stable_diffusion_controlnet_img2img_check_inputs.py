def check_inputs(self, prompt, image, controlnet_conditioning_image, height,
    width, callback_steps, negative_prompt=None, prompt_embeds=None,
    negative_prompt_embeds=None, strength=None, controlnet_guidance_start=
    None, controlnet_guidance_end=None, controlnet_conditioning_scale=None):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
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
    if isinstance(self.controlnet, ControlNetModel):
        self.check_controlnet_conditioning_image(controlnet_conditioning_image,
            prompt, prompt_embeds)
    elif isinstance(self.controlnet, MultiControlNetModel):
        if not isinstance(controlnet_conditioning_image, list):
            raise TypeError(
                'For multiple controlnets: `image` must be type `list`')
        if len(controlnet_conditioning_image) != len(self.controlnet.nets):
            raise ValueError(
                'For multiple controlnets: `image` must have the same length as the number of controlnets.'
                )
        for image_ in controlnet_conditioning_image:
            self.check_controlnet_conditioning_image(image_, prompt,
                prompt_embeds)
    else:
        assert False
    if isinstance(self.controlnet, ControlNetModel):
        if not isinstance(controlnet_conditioning_scale, float):
            raise TypeError(
                'For single controlnet: `controlnet_conditioning_scale` must be type `float`.'
                )
    elif isinstance(self.controlnet, MultiControlNetModel):
        if isinstance(controlnet_conditioning_scale, list) and len(
            controlnet_conditioning_scale) != len(self.controlnet.nets):
            raise ValueError(
                'For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have the same length as the number of controlnets'
                )
    else:
        assert False
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 and image.ndim != 4:
            raise ValueError('`image` must have 3 or 4 dimensions')
        if image.ndim == 3:
            image_batch_size = 1
            image_channels, image_height, image_width = image.shape
        elif image.ndim == 4:
            image_batch_size, image_channels, image_height, image_width = (
                image.shape)
        else:
            assert False
        if image_channels != 3:
            raise ValueError('`image` must have 3 channels')
        if image.min() < -1 or image.max() > 1:
            raise ValueError('`image` should be in range [-1, 1]')
    if self.vae.config.latent_channels != self.unet.config.in_channels:
        raise ValueError(
            f'The config of `pipeline.unet` expects {self.unet.config.in_channels} but received latent channels: {self.vae.config.latent_channels}, Please verify the config of `pipeline.unet` and the `pipeline.vae`'
            )
    if strength < 0 or strength > 1:
        raise ValueError(
            f'The value of `strength` should in [0.0, 1.0] but is {strength}')
    if controlnet_guidance_start < 0 or controlnet_guidance_start > 1:
        raise ValueError(
            f'The value of `controlnet_guidance_start` should in [0.0, 1.0] but is {controlnet_guidance_start}'
            )
    if controlnet_guidance_end < 0 or controlnet_guidance_end > 1:
        raise ValueError(
            f'The value of `controlnet_guidance_end` should in [0.0, 1.0] but is {controlnet_guidance_end}'
            )
    if controlnet_guidance_start > controlnet_guidance_end:
        raise ValueError(
            f'The value of `controlnet_guidance_start` should be less than `controlnet_guidance_end`, but got `controlnet_guidance_start` {controlnet_guidance_start} >= `controlnet_guidance_end` {controlnet_guidance_end}'
            )
