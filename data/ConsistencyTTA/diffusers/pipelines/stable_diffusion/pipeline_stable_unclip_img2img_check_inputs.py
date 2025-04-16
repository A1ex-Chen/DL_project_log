def check_inputs(self, prompt, image, height, width, callback_steps,
    noise_level, negative_prompt=None, prompt_embeds=None,
    negative_prompt_embeds=None, image_embeds=None):
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
            'Provide either `prompt` or `prompt_embeds`. Please make sure to define only one of the two.'
            )
    if prompt is None and prompt_embeds is None:
        raise ValueError(
            'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
            )
    if prompt is not None and (not isinstance(prompt, str) and not
        isinstance(prompt, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            'Provide either `negative_prompt` or `negative_prompt_embeds`. Cannot leave both `negative_prompt` and `negative_prompt_embeds` undefined.'
            )
    if prompt is not None and negative_prompt is not None:
        if type(prompt) is not type(negative_prompt):
            raise TypeError(
                f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )
    if (noise_level < 0 or noise_level >= self.image_noising_scheduler.
        config.num_train_timesteps):
        raise ValueError(
            f'`noise_level` must be between 0 and {self.image_noising_scheduler.config.num_train_timesteps - 1}, inclusive.'
            )
    if image is not None and image_embeds is not None:
        raise ValueError(
            'Provide either `image` or `image_embeds`. Please make sure to define only one of the two.'
            )
    if image is None and image_embeds is None:
        raise ValueError(
            'Provide either `image` or `image_embeds`. Cannot leave both `image` and `image_embeds` undefined.'
            )
    if image is not None:
        if not isinstance(image, torch.Tensor) and not isinstance(image,
            PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(
                f'`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is {type(image)}'
                )
