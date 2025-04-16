def test_openai_gpt_model(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_openai_gpt_model(*config_and_inputs)
