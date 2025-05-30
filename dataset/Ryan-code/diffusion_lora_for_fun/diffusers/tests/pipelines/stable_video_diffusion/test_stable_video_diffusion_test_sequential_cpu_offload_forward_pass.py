@unittest.skipIf(torch_device != 'cuda' or not is_accelerate_available() or
    is_accelerate_version('<', '0.14.0'), reason=
    'CPU offload is only available with CUDA and `accelerate v0.14.0` or higher'
    )
def test_sequential_cpu_offload_forward_pass(self, expected_max_diff=0.0001):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    inputs = self.get_dummy_inputs(generator_device)
    output_without_offload = pipe(**inputs).frames[0]
    pipe.enable_sequential_cpu_offload()
    inputs = self.get_dummy_inputs(generator_device)
    output_with_offload = pipe(**inputs).frames[0]
    max_diff = np.abs(to_np(output_with_offload) - to_np(
        output_without_offload)).max()
    self.assertLess(max_diff, expected_max_diff,
        'CPU offloading should not affect the inference results')
