def test_gpt2_double_head(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_gpt2_double_head(*config_and_inputs)
