def test_initialization(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    configs_no_init = _config_zero_init(config)
    for model_class in self.all_model_classes:
        model = model_class(config=configs_no_init)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIn(param.data.mean().item(), [0.0, 1.0], msg=
                    'Parameter {} of model {} seems not properly initialized'
                    .format(name, model_class))
