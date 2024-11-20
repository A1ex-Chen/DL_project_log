@unittest.skipIf(torch_device != 'cuda' or not is_accelerate_available() or
    is_accelerate_version('<', '0.14.0'), reason=
    'CPU offload is only available with CUDA and `accelerate v0.14.0` or higher'
    )
def test_from_pipe_consistent_forward_pass_cpu_offload(self,
    expected_max_diff=0.001):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs_pipe(torch_device)
    output = pipe(**inputs)[0]
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
    pipe_original.set_progress_bar_config(disable=None)
    pipe_from_original = self.pipeline_class.from_pipe(pipe_original, **
        current_pipe_additional_components)
    pipe_from_original.enable_model_cpu_offload()
    pipe_from_original.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs_pipe(torch_device)
    output_from_original = pipe_from_original(**inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_from_original)).max()
    self.assertLess(max_diff, expected_max_diff,
        'The outputs of the pipelines created with `from_pipe` and `__init__` are different.'
        )
