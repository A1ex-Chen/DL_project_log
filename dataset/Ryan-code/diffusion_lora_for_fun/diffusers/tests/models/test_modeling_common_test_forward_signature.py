def test_forward_signature(self):
    init_dict, _ = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    signature = inspect.signature(model.forward)
    arg_names = [*signature.parameters.keys()]
    expected_arg_names = ['sample', 'timestep']
    self.assertListEqual(arg_names[:2], expected_arg_names)
