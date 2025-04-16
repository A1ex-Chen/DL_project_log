def check_inputs(self, prompt, image, batch_size, callback_steps,
    negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
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
    if isinstance(image, list):
        check_image_type = image[0]
    else:
        check_image_type = image
    if not isinstance(check_image_type, torch.Tensor) and not isinstance(
        check_image_type, PIL.Image.Image) and not isinstance(check_image_type,
        np.ndarray):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is {type(check_image_type)}'
            )
    if isinstance(image, list):
        image_batch_size = len(image)
    elif isinstance(image, torch.Tensor):
        image_batch_size = image.shape[0]
    elif isinstance(image, PIL.Image.Image):
        image_batch_size = 1
    elif isinstance(image, np.ndarray):
        image_batch_size = image.shape[0]
    else:
        assert False
    if batch_size != image_batch_size:
        raise ValueError(
            f'image batch size: {image_batch_size} must be same as prompt batch size {batch_size}'
            )
