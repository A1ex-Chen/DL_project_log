def test_from_pipe_consistent_forward_pass(self, expected_max_diff=0.001):
    components = self.get_dummy_components()
    original_expected_modules, _ = (self.original_pipeline_class.
        _get_signature_keys(self.original_pipeline_class))
    original_pipe_components = {}
    original_pipe_additional_components = {}
    current_pipe_additional_components = {}
    for name, component in components.items():
        if name in original_expected_modules:
            original_pipe_components[name] = component
        else:
            current_pipe_additional_components[name] = component
    for name in original_expected_modules:
        if name not in original_pipe_components:
            if name in self.original_pipeline_class._optional_components:
                original_pipe_additional_components[name] = None
            else:
                raise ValueError(
                    f'missing required module for {self.original_pipeline_class.__class__}: {name}'
                    )
    pipe_original = self.original_pipeline_class(**original_pipe_components,
        **original_pipe_additional_components)
    for component in pipe_original.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe_original.to(torch_device)
    pipe_original.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs_for_pipe_original(torch_device)
    output_original = pipe_original(**inputs)[0]
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs_pipe(torch_device)
    output = pipe(**inputs)[0]
    pipe_from_original = self.pipeline_class.from_pipe(pipe_original, **
        current_pipe_additional_components)
    pipe_from_original.to(torch_device)
    pipe_from_original.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs_pipe(torch_device)
    output_from_original = pipe_from_original(**inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_from_original)).max()
    self.assertLess(max_diff, expected_max_diff,
        'The outputs of the pipelines created with `from_pipe` and `__init__` are different.'
        )
    inputs = self.get_dummy_inputs_for_pipe_original(torch_device)
    output_original_2 = pipe_original(**inputs)[0]
    max_diff = np.abs(to_np(output_original) - to_np(output_original_2)).max()
    self.assertLess(max_diff, expected_max_diff,
        '`from_pipe` should not change the output of original pipeline.')
    for component in pipe_original.components.values():
        if hasattr(component, 'attn_processors'):
            assert all(type(proc) == AttnProcessor for proc in component.
                attn_processors.values()
                ), '`from_pipe` changed the attention processor in original pipeline.'
