def test_xlnet_lm_head(self):
    self.model_tester.set_seed()
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_xlnet_lm_head(*config_and_inputs)
