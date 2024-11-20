def test_xlm_simple_qa(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_xlm_simple_qa(*config_and_inputs)
