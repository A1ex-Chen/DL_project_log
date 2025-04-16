def test_forward_with_norm_groups(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['norm_num_groups'] = 16
    init_dict['block_out_channels'] = 16, 32
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        output = model(**inputs_dict)
        if isinstance(output, dict):
            output = output.to_tuple()[0]
    self.assertIsNotNone(output)
    expected_shape = inputs_dict['sample'].shape
    self.assertEqual(output.shape, expected_shape,
        'Input and output shapes do not match')
