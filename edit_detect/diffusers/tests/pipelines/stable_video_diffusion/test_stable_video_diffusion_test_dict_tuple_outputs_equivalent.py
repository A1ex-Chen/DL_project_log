def test_dict_tuple_outputs_equivalent(self, expected_max_difference=0.0001):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    output = pipe(**self.get_dummy_inputs(generator_device)).frames[0]
    output_tuple = pipe(**self.get_dummy_inputs(generator_device),
        return_dict=False)[0]
    max_diff = np.abs(to_np(output) - to_np(output_tuple)).max()
    self.assertLess(max_diff, expected_max_difference)
