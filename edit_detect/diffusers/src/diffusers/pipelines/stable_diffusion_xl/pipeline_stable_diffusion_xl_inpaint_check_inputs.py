def check_inputs(self, prompt, prompt_2, image, mask_image, height, width,
    strength, callback_steps, output_type, negative_prompt=None,
    negative_prompt_2=None, prompt_embeds=None, negative_prompt_embeds=None,
    ip_adapter_image=None, ip_adapter_image_embeds=None,
    callback_on_step_end_tensor_inputs=None, padding_mask_crop=None):
    if strength < 0 or strength > 1:
        raise ValueError(
            f'The value of strength should in [0.0, 1.0] but is {strength}')
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    if callback_steps is not None and (not isinstance(callback_steps, int) or
        callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    if callback_on_step_end_tensor_inputs is not None and not all(k in self
        ._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
        raise ValueError(
            f'`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}'
            )
    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
            )
    elif prompt_2 is not None and prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
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
    elif prompt_2 is not None and (not isinstance(prompt_2, str) and not
        isinstance(prompt_2, list)):
        raise ValueError(
            f'`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}'
            )
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
    elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )
    if padding_mask_crop is not None:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(
                f'The image should be a PIL image when inpainting mask crop, but is of type {type(image)}.'
                )
        if not isinstance(mask_image, PIL.Image.Image):
            raise ValueError(
                f'The mask image should be a PIL image when inpainting mask crop, but is of type {type(mask_image)}.'
                )
        if output_type != 'pil':
            raise ValueError(
                f'The output type should be PIL when inpainting mask crop, but is {output_type}.'
                )
    if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
        raise ValueError(
            'Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined.'
            )
    if ip_adapter_image_embeds is not None:
        if not isinstance(ip_adapter_image_embeds, list):
            raise ValueError(
                f'`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}'
                )
        elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
            raise ValueError(
                f'`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D'
                )
