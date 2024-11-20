@classmethod
def from_pipe(cls, pipeline, **kwargs):
    """
        Create a new pipeline from a given pipeline. This method is useful to create a new pipeline from the existing
        pipeline components without reallocating additional memory.

        Arguments:
            pipeline (`DiffusionPipeline`):
                The pipeline from which to create a new pipeline.

        Returns:
            `DiffusionPipeline`:
                A new pipeline with the same weights and configurations as `pipeline`.

        Examples:

        ```py
        >>> from diffusers import StableDiffusionPipeline, StableDiffusionSAGPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> new_pipe = StableDiffusionSAGPipeline.from_pipe(pipe)
        ```
        """
    original_config = dict(pipeline.config)
    torch_dtype = kwargs.pop('torch_dtype', None)
    custom_pipeline = kwargs.pop('custom_pipeline', None)
    custom_revision = kwargs.pop('custom_revision', None)
    if custom_pipeline is not None:
        pipeline_class = _get_custom_pipeline_class(custom_pipeline,
            revision=custom_revision)
    else:
        pipeline_class = cls
    expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
    parameters = inspect.signature(cls.__init__).parameters
    true_optional_modules = set({k for k, v in parameters.items() if v.
        default != inspect._empty and k in expected_modules})
    component_types = pipeline_class._get_signature_types()
    pretrained_model_name_or_path = original_config.pop('_name_or_path', None)
    passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in
        kwargs}
    original_class_obj = {}
    for name, component in pipeline.components.items():
        if name in expected_modules and name not in passed_class_obj:
            if not isinstance(component, ModelMixin) or type(component
                ) in component_types[name
                ] or component is None and name in cls._optional_components:
                original_class_obj[name] = component
            else:
                logger.warning(
                    f'component {name} is not switched over to new pipeline because type does not match the expected. {name} is {type(component)} while the new pipeline expect {component_types[name]}. please pass the component of the correct type to the new pipeline. `from_pipe(..., {name}={name})`'
                    )
    passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in
        kwargs}
    original_pipe_kwargs = {k: original_config[k] for k in original_config.
        keys() if k in optional_kwargs and k not in passed_pipe_kwargs}
    additional_pipe_kwargs = [k[1:] for k in original_config.keys() if k.
        startswith('_') and k[1:] in optional_kwargs and k[1:] not in
        passed_pipe_kwargs]
    for k in additional_pipe_kwargs:
        original_pipe_kwargs[k] = original_config.pop(f'_{k}')
    pipeline_kwargs = {**passed_class_obj, **original_class_obj, **
        passed_pipe_kwargs, **original_pipe_kwargs, **kwargs}
    unused_original_config = {f"{'' if k.startswith('_') else '_'}{k}": v for
        k, v in original_config.items() if k not in pipeline_kwargs}
    missing_modules = set(expected_modules) - set(pipeline._optional_components
        ) - set(pipeline_kwargs.keys()) - set(true_optional_modules)
    if len(missing_modules) > 0:
        raise ValueError(
            f'Pipeline {pipeline_class} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed'
            )
    new_pipeline = pipeline_class(**pipeline_kwargs)
    if pretrained_model_name_or_path is not None:
        new_pipeline.register_to_config(_name_or_path=
            pretrained_model_name_or_path)
    new_pipeline.register_to_config(**unused_original_config)
    if torch_dtype is not None:
        new_pipeline.to(dtype=torch_dtype)
    return new_pipeline
