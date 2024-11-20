def test_output(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        output = model(**inputs_dict)
        if isinstance(output, dict):
            output = output.to_tuple()[0]
    self.assertIsNotNone(output)
    input_tensor = inputs_dict[self.main_input_name]
    expected_shape = input_tensor.shape
    self.assertEqual(output.shape, expected_shape,
        'Input and output shapes do not match')
