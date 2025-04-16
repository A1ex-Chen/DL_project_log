def check_conditions(self, prompt, prompt_embeds, adapter_image,
    control_image, adapter_conditioning_scale,
    controlnet_conditioning_scale, control_guidance_start, control_guidance_end
    ):
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
    is_compiled = hasattr(F, 'scaled_dot_product_attention') and isinstance(
        self.controlnet, torch._dynamo.eval_frame.OptimizedModule)
    if isinstance(self.controlnet, ControlNetModel
        ) or is_compiled and isinstance(self.controlnet._orig_mod,
        ControlNetModel):
        self.check_image(control_image, prompt, prompt_embeds)
    elif isinstance(self.controlnet, MultiControlNetModel
        ) or is_compiled and isinstance(self.controlnet._orig_mod,
        MultiControlNetModel):
        if not isinstance(control_image, list):
            raise TypeError(
                'For multiple controlnets: `control_image` must be type `list`'
                )
        elif any(isinstance(i, list) for i in control_image):
            raise ValueError(
                'A single batch of multiple conditionings are supported at the moment.'
                )
        elif len(control_image) != len(self.controlnet.nets):
            raise ValueError(
                f'For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(control_image)} images and {len(self.controlnet.nets)} ControlNets.'
                )
        for image_ in control_image:
            self.check_image(image_, prompt, prompt_embeds)
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
    if isinstance(self.adapter, T2IAdapter) or is_compiled and isinstance(self
        .adapter._orig_mod, T2IAdapter):
        self.check_image(adapter_image, prompt, prompt_embeds)
    elif isinstance(self.adapter, MultiAdapter) or is_compiled and isinstance(
        self.adapter._orig_mod, MultiAdapter):
        if not isinstance(adapter_image, list):
            raise TypeError(
                'For multiple adapters: `adapter_image` must be type `list`')
        elif any(isinstance(i, list) for i in adapter_image):
            raise ValueError(
                'A single batch of multiple conditionings are supported at the moment.'
                )
        elif len(adapter_image) != len(self.adapter.adapters):
            raise ValueError(
                f'For multiple adapters: `image` must have the same length as the number of adapters, but got {len(adapter_image)} images and {len(self.adapters.nets)} Adapters.'
                )
        for image_ in adapter_image:
            self.check_image(image_, prompt, prompt_embeds)
    else:
        assert False
    if isinstance(self.adapter, T2IAdapter) or is_compiled and isinstance(self
        .adapter._orig_mod, T2IAdapter):
        if not isinstance(adapter_conditioning_scale, float):
            raise TypeError(
                'For single adapter: `adapter_conditioning_scale` must be type `float`.'
                )
    elif isinstance(self.adapter, MultiAdapter) or is_compiled and isinstance(
        self.adapter._orig_mod, MultiAdapter):
        if isinstance(adapter_conditioning_scale, list):
            if any(isinstance(i, list) for i in adapter_conditioning_scale):
                raise ValueError(
                    'A single batch of multiple conditionings are supported at the moment.'
                    )
        elif isinstance(adapter_conditioning_scale, list) and len(
            adapter_conditioning_scale) != len(self.adapter.adapters):
            raise ValueError(
                'For multiple adapters: When `adapter_conditioning_scale` is specified as `list`, it must have the same length as the number of adapters'
                )
    else:
        assert False
