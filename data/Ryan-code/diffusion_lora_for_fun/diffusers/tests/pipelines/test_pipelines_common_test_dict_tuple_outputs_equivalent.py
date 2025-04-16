def test_dict_tuple_outputs_equivalent(self, expected_slice=None,
    expected_max_difference=0.0001):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    if expected_slice is None:
        output = pipe(**self.get_dummy_inputs(generator_device))[0]
    else:
        output = expected_slice
    output_tuple = pipe(**self.get_dummy_inputs(generator_device),
        return_dict=False)[0]
    if expected_slice is None:
        max_diff = np.abs(to_np(output) - to_np(output_tuple)).max()
    elif output_tuple.ndim != 5:
        max_diff = np.abs(to_np(output) - to_np(output_tuple)[0, -3:, -3:, 
            -1].flatten()).max()
    else:
        max_diff = np.abs(to_np(output) - to_np(output_tuple)[0, -3:, -3:, 
            -1, -1].flatten()).max()
    self.assertLess(max_diff, expected_max_difference)
