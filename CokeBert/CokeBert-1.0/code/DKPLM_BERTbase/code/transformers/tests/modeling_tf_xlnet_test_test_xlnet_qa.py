def test_xlnet_qa(self):
    self.model_tester.set_seed()
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_xlnet_qa(*config_and_inputs)
