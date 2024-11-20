def test_from_pipe_consistent_config(self):
    if self.original_pipeline_class == StableDiffusionPipeline:
        original_repo = 'hf-internal-testing/tiny-stable-diffusion-pipe'
        original_kwargs = {'requires_safety_checker': False}
    elif self.original_pipeline_class == StableDiffusionXLPipeline:
        original_repo = 'hf-internal-testing/tiny-stable-diffusion-xl-pipe'
        original_kwargs = {'requires_aesthetics_score': True,
            'force_zeros_for_empty_prompt': False}
    else:
        raise ValueError(
            'original_pipeline_class must be either StableDiffusionPipeline or StableDiffusionXLPipeline'
            )
    pipe_original = self.original_pipeline_class.from_pretrained(original_repo,
        **original_kwargs)
    pipe_components = self.get_dummy_components()
    pipe_additional_components = {}
    for name, component in pipe_components.items():
        if name not in pipe_original.components:
            pipe_additional_components[name] = component
    pipe = self.pipeline_class.from_pipe(pipe_original, **
        pipe_additional_components)
    original_pipe_additional_components = {}
    for name, component in pipe_original.components.items():
        if name not in pipe.components or not isinstance(component, pipe.
            components[name].__class__):
            original_pipe_additional_components[name] = component
    pipe_original_2 = self.original_pipeline_class.from_pipe(pipe, **
        original_pipe_additional_components)
    original_config = {k: v for k, v in pipe_original.config.items() if not
        k.startswith('_')}
    original_config_2 = {k: v for k, v in pipe_original_2.config.items() if
        not k.startswith('_')}
    assert original_config_2 == original_config
