def check_inputs(self, prompt, height, width, num_frames, callback_steps,
    negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
    callback_on_step_end_tensor_inputs=None, image=None,
    controlnet_conditioning_scale=1.0, control_guidance_start=0.0,
    control_guidance_end=1.0):
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
    is_compiled = hasattr(F, 'scaled_dot_product_attention') and isinstance(
        self.controlnet, torch._dynamo.eval_frame.OptimizedModule)
    if isinstance(self.controlnet, ControlNetModel
        ) or is_compiled and isinstance(self.controlnet._orig_mod,
        ControlNetModel):
        if not isinstance(image, list):
            raise TypeError(
                f'For single controlnet, `image` must be of type `list` but got {type(image)}'
                )
        if len(image) != num_frames:
            raise ValueError(
                f'Excepted image to have length {num_frames} but got len(image)={len(image)!r}'
                )
    elif isinstance(self.controlnet, MultiControlNetModel
        ) or is_compiled and isinstance(self.controlnet._orig_mod,
        MultiControlNetModel):
        if not isinstance(image, list) or not isinstance(image[0], list):
            raise TypeError(
                f'For multiple controlnets: `image` must be type list of lists but got type(image)={type(image)!r}'
                )
        if len(image[0]) != num_frames:
            raise ValueError(
                f'Expected length of image sublist as {num_frames} but got len(image[0])={len(image[0])!r}'
                )
        if any(len(img) != len(image[0]) for img in image):
            raise ValueError(
                'All conditioning frame batches for multicontrolnet must be same size'
                )
    else:
        assert False
    if isinstance(self.controlnet, ControlNetModel
        ) or is_compiled and isinstance(self.controlnet._orig_mod,
        ControlNetModel):
        if not isinstance(controlnet_conditioning_scale, float):
            raise TypeError(
                'For single controlnet: `controlnet_conditioning_scale` must be type `float`.'
                )
    elif isinstance(self.controlnet, MultiControlNetModel
        ) or is_compiled and isinstance(self.controlnet._orig_mod,
        MultiControlNetModel):
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
    if not isinstance(control_guidance_start, (tuple, list)):
        control_guidance_start = [control_guidance_start]
    if not isinstance(control_guidance_end, (tuple, list)):
        control_guidance_end = [control_guidance_end]
    if len(control_guidance_start) != len(control_guidance_end):
        raise ValueError(
            f'`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list.'
            )
    if isinstance(self.controlnet, MultiControlNetModel):
        if len(control_guidance_start) != len(self.controlnet.nets):
            raise ValueError(
                f'`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}.'
                )
    for start, end in zip(control_guidance_start, control_guidance_end):
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