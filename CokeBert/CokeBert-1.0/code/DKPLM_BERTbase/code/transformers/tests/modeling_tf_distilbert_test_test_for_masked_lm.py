def test_for_masked_lm(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_distilbert_for_masked_lm(*
        config_and_inputs)
