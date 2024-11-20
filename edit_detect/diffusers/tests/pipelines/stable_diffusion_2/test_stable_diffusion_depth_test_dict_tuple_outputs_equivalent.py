def test_dict_tuple_outputs_equivalent(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(torch_device))[0]
    output_tuple = pipe(**self.get_dummy_inputs(torch_device), return_dict=
        False)[0]
    max_diff = np.abs(output - output_tuple).max()
    self.assertLess(max_diff, 0.0001)
