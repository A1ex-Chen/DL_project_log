def check_inputs(self, prompt, images=None, image_embeds=None,
    negative_prompt=None, prompt_embeds=None, prompt_embeds_pooled=None,
    negative_prompt_embeds=None, negative_prompt_embeds_pooled=None,
    callback_on_step_end_tensor_inputs=None):
    if callback_on_step_end_tensor_inputs is not None and not all(k in self
        ._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
        raise ValueError(
            f'`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}'
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
    if prompt_embeds is not None and prompt_embeds_pooled is None:
        raise ValueError(
            'If `prompt_embeds` are provided, `prompt_embeds_pooled` must also be provided. Make sure to generate `prompt_embeds_pooled` from the same text encoder that was used to generate `prompt_embeds`'
            )
    if (negative_prompt_embeds is not None and 
        negative_prompt_embeds_pooled is None):
        raise ValueError(
            'If `negative_prompt_embeds` are provided, `negative_prompt_embeds_pooled` must also be provided. Make sure to generate `prompt_embeds_pooled` from the same text encoder that was used to generate `prompt_embeds`'
            )
    if (prompt_embeds_pooled is not None and negative_prompt_embeds_pooled
         is not None):
        if prompt_embeds_pooled.shape != negative_prompt_embeds_pooled.shape:
            raise ValueError(
                f'`prompt_embeds_pooled` and `negative_prompt_embeds_pooled` must have the same shape when passeddirectly, but got: `prompt_embeds_pooled` {prompt_embeds_pooled.shape} !=`negative_prompt_embeds_pooled` {negative_prompt_embeds_pooled.shape}.'
                )
    if image_embeds is not None and images is not None:
        raise ValueError(
            f'Cannot forward both `images`: {images} and `image_embeds`: {image_embeds}. Please make sure to only forward one of the two.'
            )
    if images:
        for i, image in enumerate(images):
            if not isinstance(image, torch.Tensor) and not isinstance(image,
                PIL.Image.Image):
                raise TypeError(
                    f"'images' must contain images of type 'torch.Tensor' or 'PIL.Image.Image, but got{type(image)} for image number {i}."
                    )
