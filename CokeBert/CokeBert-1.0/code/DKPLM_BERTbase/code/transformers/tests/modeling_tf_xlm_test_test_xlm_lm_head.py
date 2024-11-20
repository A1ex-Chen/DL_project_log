def test_xlm_lm_head(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_xlm_lm_head(*config_and_inputs)
