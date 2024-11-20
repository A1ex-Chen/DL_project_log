def test_pt_np_pil_inputs_equivalent(self):
    if len(self.image_params) == 0:
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    out_input_pt = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type='pt'))[0]
    out_input_np = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type='np'))[0]
    out_input_pil = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type='pil'))[0]
    max_diff = np.abs(out_input_pt - out_input_np).max()
    self.assertLess(max_diff, 0.0001,
        "`input_type=='pt'` generate different result from `input_type=='np'`")
    max_diff = np.abs(out_input_pil - out_input_np).max()
    self.assertLess(max_diff, 0.01,
        "`input_type=='pt'` generate different result from `input_type=='np'`")
