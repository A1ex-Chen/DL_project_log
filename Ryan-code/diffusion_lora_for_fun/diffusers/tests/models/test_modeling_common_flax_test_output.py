def test_output(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    variables = model.init(inputs_dict['prng_key'], inputs_dict['sample'])
    jax.lax.stop_gradient(variables)
    output = model.apply(variables, inputs_dict['sample'])
    if isinstance(output, dict):
        output = output.sample
    self.assertIsNotNone(output)
    expected_shape = inputs_dict['sample'].shape
    self.assertEqual(output.shape, expected_shape,
        'Input and output shapes do not match')
