def test_from_pretrained_hub(self):
    value_function, vf_loading_info = UNet1DModel.from_pretrained(
        'bglick13/hopper-medium-v2-value-function-hor32',
        output_loading_info=True, subfolder='value_function')
    self.assertIsNotNone(value_function)
    self.assertEqual(len(vf_loading_info['missing_keys']), 0)
    value_function.to(torch_device)
    image = value_function(**self.dummy_input)
    assert image is not None, 'Make sure output is not None'
