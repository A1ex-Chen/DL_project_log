def test_dict_tuple_outputs_equivalent(self, expected_max_difference=0.0001):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(self.generator_device))[0]
    output_tuple = pipe(**self.get_dummy_inputs(self.generator_device),
        return_dict=False)[0]
    max_diff = np.abs(to_np(output) - to_np(output_tuple)).max()
    self.assertLess(max_diff, expected_max_difference)
