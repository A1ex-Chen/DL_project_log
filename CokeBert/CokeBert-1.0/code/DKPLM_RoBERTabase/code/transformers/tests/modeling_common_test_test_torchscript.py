def test_torchscript(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    self._create_and_check_torchscript(config, inputs_dict)
