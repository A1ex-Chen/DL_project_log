def _test_attention_slicing_forward_pass(self, test_max_difference=True,
    test_mean_pixel_difference=True, expected_max_diff=0.001):
    if not self.test_attention_slicing:
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    inputs = self.get_dummy_inputs(generator_device)
    output_without_slicing = pipe(**inputs)[0]
    pipe.enable_attention_slicing(slice_size=1)
    inputs = self.get_dummy_inputs(generator_device)
    output_with_slicing = pipe(**inputs)[0]
    if test_max_difference:
        max_diff = np.abs(to_np(output_with_slicing) - to_np(
            output_without_slicing)).max()
        self.assertLess(max_diff, expected_max_diff,
            'Attention slicing should not affect the inference results')
    if test_mean_pixel_difference:
        assert_mean_pixel_difference(to_np(output_with_slicing[0]), to_np(
            output_without_slicing[0]))
