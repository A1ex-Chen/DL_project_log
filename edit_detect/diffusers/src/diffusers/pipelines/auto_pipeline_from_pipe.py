@classmethod
def from_pipe(cls, pipeline, **kwargs):
    """
        Instantiates a inpainting Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the inpainting
        pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline class contain will be used to initialize the new pipeline without reallocating
        additional memory.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting

        >>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0", requires_safety_checker=False
        ... )

        >>> pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_t2i)
        >>> image = pipe_inpaint(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
    original_config = dict(pipeline.config)
    original_cls_name = pipeline.__class__.__name__
    inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING,
        original_cls_name)
    if 'controlnet' in kwargs:
        if kwargs['controlnet'] is not None:
            inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING,
                inpainting_cls.__name__.replace('ControlNet', '').replace(
                'InpaintPipeline', 'ControlNetInpaintPipeline'))
        else:
            inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING,
                inpainting_cls.__name__.replace('ControlNetInpaintPipeline',
                'InpaintPipeline'))
    expected_modules, optional_kwargs = inpainting_cls._get_signature_keys(
        inpainting_cls)
    pretrained_model_name_or_path = original_config.pop('_name_or_path', None)
    passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in
        kwargs}
    original_class_obj = {k: pipeline.components[k] for k, v in pipeline.
        components.items() if k in expected_modules and k not in
        passed_class_obj}
    passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in
        kwargs}
    original_pipe_kwargs = {k: original_config[k] for k, v in
        original_config.items() if k in optional_kwargs and k not in
        passed_pipe_kwargs}
    additional_pipe_kwargs = [k[1:] for k in original_config.keys() if k.
        startswith('_') and k[1:] in optional_kwargs and k[1:] not in
        passed_pipe_kwargs]
    for k in additional_pipe_kwargs:
        original_pipe_kwargs[k] = original_config.pop(f'_{k}')
    inpainting_kwargs = {**passed_class_obj, **original_class_obj, **
        passed_pipe_kwargs, **original_pipe_kwargs}
    unused_original_config = {f"{'' if k.startswith('_') else '_'}{k}":
        original_config[k] for k, v in original_config.items() if k not in
        inpainting_kwargs}
    missing_modules = set(expected_modules) - set(pipeline._optional_components
        ) - set(inpainting_kwargs.keys())
    if len(missing_modules) > 0:
        raise ValueError(
            f'Pipeline {inpainting_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed'
            )
    model = inpainting_cls(**inpainting_kwargs)
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    model.register_to_config(**unused_original_config)
    return model
