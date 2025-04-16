def test_gpt2_lm_head_model(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_lm_head_model(*config_and_inputs)
