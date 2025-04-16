def check_inputs(self, prompt, image, height, width, callback_steps,
    negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
    controlnet_conditioning_scale=1.0):
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
    if isinstance(self.controlnet, MultiControlNetModel):
        if isinstance(prompt, list):
            logger.warning(
                f'You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)} prompts. The conditionings will be fixed across the prompts.'
                )
    if isinstance(self.controlnet, ControlNetModel):
        self.check_image(image, prompt, prompt_embeds)
    elif isinstance(self.controlnet, MultiControlNetModel):
        if not isinstance(image, list):
            raise TypeError(
                'For multiple controlnets: `image` must be type `list`')
        elif any(isinstance(i, list) for i in image):
            raise ValueError(
                'A single batch of multiple conditionings are supported at the moment.'
                )
        elif len(image) != len(self.controlnet.nets):
            raise ValueError(
                'For multiple controlnets: `image` must have the same length as the number of controlnets.'
                )
        for image_ in image:
            self.check_image(image_, prompt, prompt_embeds)
    else:
        assert False
    if isinstance(self.controlnet, ControlNetModel):
        if not isinstance(controlnet_conditioning_scale, float):
            raise TypeError(
                'For single controlnet: `controlnet_conditioning_scale` must be type `float`.'
                )
    elif isinstance(self.controlnet, MultiControlNetModel):
        if isinstance(controlnet_conditioning_scale, list):
            if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                raise ValueError(
                    'A single batch of multiple conditionings are supported at the moment.'
                    )
        elif isinstance(controlnet_conditioning_scale, list) and len(
            controlnet_conditioning_scale) != len(self.controlnet.nets):
            raise ValueError(
                'For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have the same length as the number of controlnets'
                )
    else:
        assert False
