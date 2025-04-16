def check_inputs(self, prompt, prompt_2, image, negative_prompt=None,
    negative_prompt_2=None, prompt_embeds=None, negative_prompt_embeds=None,
    pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None,
    controlnet_conditioning_scale=1.0, control_guidance_start=0.0,
    control_guidance_end=1.0, callback_on_step_end_tensor_inputs=None):
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
    if prompt_embeds is not None and pooled_prompt_embeds is None:
        raise ValueError(
            'If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`.'
            )
    if (negative_prompt_embeds is not None and 
        negative_pooled_prompt_embeds is None):
        raise ValueError(
            'If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`.'
            )
    is_compiled = hasattr(F, 'scaled_dot_product_attention') and isinstance(
        self.unet, torch._dynamo.eval_frame.OptimizedModule)
    if isinstance(self.unet, UNetControlNetXSModel
        ) or is_compiled and isinstance(self.unet._orig_mod,
        UNetControlNetXSModel):
        self.check_image(image, prompt, prompt_embeds)
        if not isinstance(controlnet_conditioning_scale, float):
            raise TypeError(
                'For single controlnet: `controlnet_conditioning_scale` must be type `float`.'
                )
    else:
        assert False
    start, end = control_guidance_start, control_guidance_end
    if start >= end:
        raise ValueError(
            f'control guidance start: {start} cannot be larger or equal to control guidance end: {end}.'
            )
    if start < 0.0:
        raise ValueError(
            f"control guidance start: {start} can't be smaller than 0.")
    if end > 1.0:
        raise ValueError(
            f"control guidance end: {end} can't be larger than 1.0.")
