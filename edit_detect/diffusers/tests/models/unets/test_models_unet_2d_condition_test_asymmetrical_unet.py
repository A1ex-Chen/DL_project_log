def test_asymmetrical_unet(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['transformer_layers_per_block'] = [[3, 2], 1]
    init_dict['reverse_transformer_layers_per_block'] = [[3, 4], 1]
    torch.manual_seed(0)
    model = self.model_class(**init_dict)
    model.to(torch_device)
    output = model(**inputs_dict).sample
    expected_shape = inputs_dict['sample'].shape
    self.assertEqual(output.shape, expected_shape,
        'Input and output shapes do not match')
