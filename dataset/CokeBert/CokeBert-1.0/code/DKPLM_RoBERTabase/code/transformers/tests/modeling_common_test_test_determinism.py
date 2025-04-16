def test_determinism(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        model = model_class(config)
        model.eval()
        first, second = model(inputs_dict['input_ids'])[0], model(inputs_dict
            ['input_ids'])[0]
        self.assertEqual(first.ne(second).sum().item(), 0)
