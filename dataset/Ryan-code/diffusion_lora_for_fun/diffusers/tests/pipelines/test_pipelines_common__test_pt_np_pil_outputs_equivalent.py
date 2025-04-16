def _test_pt_np_pil_outputs_equivalent(self, expected_max_diff=0.0001,
    input_image_type='pt'):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    output_pt = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type=input_image_type, output_type='pt'))[0]
    output_np = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type=input_image_type, output_type='np'))[0]
    output_pil = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type=input_image_type, output_type='pil'))[0]
    max_diff = np.abs(output_pt.cpu().numpy().transpose(0, 2, 3, 1) - output_np
        ).max()
    self.assertLess(max_diff, expected_max_diff,
        "`output_type=='pt'` generate different results from `output_type=='np'`"
        )
    max_diff = np.abs(np.array(output_pil[0]) - (output_np * 255).round()).max(
        )
    self.assertLess(max_diff, 2.0,
        "`output_type=='pil'` generate different results from `output_type=='np'`"
        )
