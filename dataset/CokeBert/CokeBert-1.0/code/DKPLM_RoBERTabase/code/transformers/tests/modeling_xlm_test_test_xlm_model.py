def test_xlm_model(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_xlm_model(*config_and_inputs)
