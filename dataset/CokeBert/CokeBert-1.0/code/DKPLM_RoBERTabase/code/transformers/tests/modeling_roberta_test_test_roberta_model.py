def test_roberta_model(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_roberta_model(*config_and_inputs)
